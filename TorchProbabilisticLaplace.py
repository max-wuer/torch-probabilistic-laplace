# -------------------------------------------------
#   Author: Maximilian Würschmidt
#   Minimal example of the learning algorithm studied in my dissertation
# -------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.optim as optim
#
from torch_utils import gradients, laplace, hessian_vector_product
#
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')


def simulate_brownian_exit(N=10, M=0, dt=1e-3, radius=0.5, center=(0.5, 0.5), additional_exits=False):
    cx, cy = center
    center = torch.tensor([cx, cy], dtype=torch.float32)
    sqrt_dt = math.sqrt(dt)

    trajectories = []
    for _ in range(N):
        pos = center.clone()
        traj = [pos.clone()]
        while True:
            pos = pos + torch.randn(2) * sqrt_dt
            traj.append(pos.clone())
            if torch.norm(pos - center) > radius:
                break
        trajectories.append(torch.stack(traj))  # (K_i,2)

    exit_points = []
    if additional_exits:
        for _ in range(M):
            pos = center.clone()
            while True:
                pos = pos + torch.randn(2) * sqrt_dt
                if torch.norm(pos - center) > radius:
                    exit_points.append(pos.clone())
                    break

    return trajectories, exit_points


def simulate_uniform_exit_points(M, radius=0.5, center=(0.5, 0.5), device='cpu'):
    cx, cy = center
    center = torch.tensor([cx, cy], dtype=torch.float32, device=device)

    theta = 2 * math.pi * torch.rand(M, device=device)
    x = center[0] + radius * torch.cos(theta)
    y = center[1] + radius * torch.sin(theta)

    return torch.stack([x, y], dim=1)


# -------------------------------------------------
#   TorchNet
# -------------------------------------------------
class TorchNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, width=32, depth=3, stepsize=0.001):
        self.h = stepsize
        super().__init__()
        layers = [nn.Linear(in_dim, width, bias=True)]
        UseBias = True
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width, bias=UseBias))
        layers.append(nn.Linear(width, out_dim, bias=UseBias))
        self.layers = nn.ModuleList(layers)

        self.init_weights()

    def forward(self, x):
        for layer in self.layers[:-1]:
            #x = torch.sin(layer(x))
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

    def init_weights(self, method="normal"):
        """
            "uniform": Xavier_uniform
            "normal": Xavier_normal
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if method == "uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif method == "normal":
                    nn.init.xavier_normal_(layer.weight)
                else:
                    raise ValueError(f"Unknown init method: {method}")

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        pass


# -------------------------------------------------
#   Loss functional - boundary | dynamical (= Taylor-type probabilistic) | PINN
# -------------------------------------------------
def boundary_loss(model, exit_points_M):
    u_exit = model(exit_points_M).reshape(-1)
    return (u_exit ** 2).sum()


def dynamical_loss(model, padded, mask, use_exit=False):
    Kmax, B, D = padded.shape
    assert D == 2

    f = 0.5

    x = padded.reshape(Kmax*B, 2).clone().detach().requires_grad_(True)
    u_flat = model(x).reshape(-1)
    u = u_flat.reshape(Kmax, B)
    g_flat = torch.autograd.grad(u_flat.sum(), x, retain_graph=True, create_graph=True)[0]
    g = g_flat.reshape(Kmax, B, 2)

    # Exit-point loss
    if use_exit:
        last_indices = mask.sum(dim=0).long() - 1
        exit_us = u[last_indices, torch.arange(B)]
        L_exit = (exit_us**2).sum()
    else:
        L_exit = 0.0

    # Interior-point loss
    padded_next = torch.zeros_like(padded)
    padded_next[:-1] = padded[1:]
    dp = padded_next - padded

    u_next = torch.zeros_like(u)
    u_next[:-1] = u[1:]

    g_dot_dp = (g * dp).sum(dim=2)

    dp_flat = dp.reshape(Kmax * B, 2)
    Hdp = hessian_vector_product(u_flat, x, dp_flat)
    dp_H_dp = (Hdp * dp_flat).sum(dim=1).reshape(Kmax, B)

    laplace_u = laplace(u_flat, x).reshape(Kmax,B)

    interior_term = u_next - u + f*model.h - g_dot_dp - dp_H_dp / 2 + laplace_u * model.h / 2
    pair_mask = mask * torch.cat([mask[1:], torch.zeros(1, B, device=padded.device)], dim=0)
    L_interior = (interior_term**2 * pair_mask).sum()

    return L_exit + L_interior


def pinn_loss(model, padded, mask):
    Kmax, B, D = padded.shape
    assert D == 2

    f = 0.5

    x = padded.reshape(Kmax*B, 2).clone().detach().requires_grad_(True)

    u_flat = model(x).reshape(-1)
    u = u_flat.reshape(Kmax, B)

    # Exit-point loss
    last_indices = mask.sum(dim=0).long() - 1
    exit_us = u[last_indices, torch.arange(B)]
    L_exit = (exit_us**2).sum()

    # Interior-point loss
    laplace_u = laplace(u_flat, x).reshape(Kmax,B)
    interior_term = f + laplace_u

    pair_mask = mask * torch.cat([mask[1:], torch.zeros(1, B, device=padded.device)], dim=0)
    L_interior = (interior_term**2 * pair_mask).sum()

    return L_exit + L_interior


def train_adam(model, forward, lr_adam=1e-3, epochs_adam=[1000], new_batch=False, save_training_weights=False, plot_training=False):

    optimizer = optim.Adam(model.parameters(), lr=lr_adam)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100, min_lr=1e-5)

    trajectories, exits = simulate_brownian_exit(forward.batch_size, forward.samples_boundary, forward.stepsize_h, forward.radius, forward.center)
    # SAVE training data
    # torch.save(trajectories, "output/training_forward_paths.pt")

    padded, mask = pad_forward_paths(trajectories, forward)
    exact_exits = simulate_uniform_exit_points(forward.samples_boundary_exact, forward.radius, forward.center)

    total_epochs = sum(epochs_adam)
    epoch_counter = 0
    for num_epochs_adam in epochs_adam: # constant learning rate decay outer loop
        for epoch in range(num_epochs_adam):

            if new_batch and (epoch_counter+(epoch+1)) % forward.new_batch_iter == 0:
                trajectories, exits = simulate_brownian_exit(forward.batch_size, forward.samples_boundary,
                                                             forward.stepsize_h, forward.radius, forward.center)
                padded, mask = pad_forward_paths(trajectories, forward)
                exact_exits = simulate_uniform_exit_points(forward.samples_boundary_exact, forward.radius, forward.center)

            optimizer.zero_grad()

            loss_b_exact =  boundary_loss(model, exact_exits) # *4
            loss_d = dynamical_loss(model, padded, mask)

            loss = (forward.samples_boundary_exact * loss_b_exact + forward.num_interior * loss_d ) / (forward.samples_boundary_exact + forward.num_interior)
            loss.backward()
            optimizer.step()

            # todo this should be computed with validation batch
            #scheduler.step(loss)

            save_train_vals(epoch_counter+epoch, loss.item(), filename="output/history.txt")

            if (epoch+1) % 1000 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Adam] {epoch_counter+epoch+1}/{total_epochs}  |  Loss: {loss.item():.8f}  |  LR: {current_lr:.1e}  |  Batch Size: {forward.batch_size}   |   Steps Total: {forward.num_true}, Mean: {forward.mean_steps}, Max: {forward.max_steps}, Interior: {forward.num_interior}  |  Exact Boundary Points: {forward.samples_boundary_exact}")
            if save_training_weights and (epoch+1) % 500 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                save_model_state(model, "output/training_weights/adam_new_%d_lr_%f.pth" % (epoch+1, current_lr))
            if plot_training and epoch % 1000 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                plot_model_variable_paths(model, trajectories, exits_exact=exact_exits, title="Model trainings iteration %d_lr_%e" % (epoch_counter+epoch, current_lr), iter=epoch_counter+epoch)
                plot_model_variable_paths(model, trajectories, exits_exact=exact_exits, title="Exact_Exit_Model trainings iteration %d_lr_%e" % (epoch_counter+epoch, current_lr), iter=epoch_counter+epoch, no_forward_exit=True)

                model_path_vals_list = [model(path) for path in trajectories[0:5]]
                exact_path_vals_list = [forward.exact_solution(path) for path in trajectories[0:5]]
                plot_timelines(path_vals_list=model_path_vals_list, vals_exact=exact_path_vals_list, stepsize=forward.stepsize_h, save_as='output/plots/timelines_%d_lr' % (epoch_counter+epoch))

            tol = 1e-6
            if loss.item() < tol:
                print(f"[Loss < Tolerance] Stop Training: Loss: {loss.item():.8f}  |  Tolerance: {tol:.1e} ")
                break

        epoch_counter += num_epochs_adam
        print("[LR Update]")
        optimizer.param_groups[0]['lr'] *= 0.1

    plot_history(filename="output/history.txt", save_as="output/loss_history")
    return model

"""
def train_model_bfgs(model, padded, mask, exit_points_M=None, exact_exits_M=None,
                num_epochs_bfgs=100,
                chunk_size=8,
                max_iter=200):

    if exit_points_M is not None:
        # falls Liste von Exit-Punkten
        if isinstance(exit_points_M, list):
            exit_points_M = torch.stack([p.clone().detach().float() for p in exit_points_M], dim=0)
        else:
            exit_points_M = exit_points_M.float()

    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter, line_search_fn="strong_wolfe", history_size=1000, tolerance_change=0)

    def closure():
        optimizer_lbfgs.zero_grad()
        loss_b = boundary_loss(model, exit_points_M)
        loss_b_exact = 4 * boundary_loss(model, exact_exits_M)
        #print('eval loss')
        loss_d = dynamical_loss_loop(model, padded, mask, chunk_size=chunk_size)
        #print('####')
        loss = loss_b + loss_b_exact + loss_d
        loss.backward()
        return loss

    for epoch in range(num_epochs_bfgs):
        if False: # epoch % 1 == 0:
            coords, exits = simulate_brownian_exit(N=50)  # z.B. (500,2)
            plot_model_variable_paths(model, coords, exit_points_M, exact_exits_M, title="Model trainings iteratrion %d" % epoch)

        loss = optimizer_lbfgs.step(closure)
        print(f"[L-BFGS] Epoch {epoch+1}/{num_epochs_bfgs}, Loss: {loss.item():.6f}")

        if (epoch+1) % 1 == 0:
            save_model_state(model, "output/training_states/sinnet_weights_bfgs_new_%d.pth" % (epoch+1) )

    return model
"""


# -------------------------------------------------
def save_model_state(model, filepath):
    torch.save(model.state_dict(), filepath)
    pass


def load_model_state(model_class, filepath, **model_kwargs):
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(filepath))
    return model


def save_train_vals(iter, *values, filename="history.txt"):
    mode = "w" if iter == 0 else "a"
    line = ";".join(str(v) for v in values)
    with open(filename, mode) as f:
        f.write(line + "\n")
    pass
# -------------------------------------------------


# -------------------------------------------------
#   plotting
#   old plotting method  called during training - not active at the moment. might need debugging
def plot_model_variable_paths(model, paths, exits=None, exits_exact=None, title="Model Output Scatterplot", figsize=(6, 6), cmap="viridis", iter=0, no_forward_exit=False):
    """
    paths : list von Pfaden, jeder Pfad (K_i,2), kann Tensor oder Liste sein
    model : torch.nn.Module, gibt 1D Tensor pro Punkt zurück

    Plottet alle Punkte in einem Scatterplot, Farbe nach Model-Output
    """

    c_min, c_max = 0.0, 0.0

    coords_list = []
    for path in paths:
        # Falls path schon Tensor ist

        if isinstance(path, torch.Tensor):
            path_tensor = path.clone().detach().float()  # 🔹 clone.detach statt torch.tensor()
        else:
            path_tensor = torch.tensor(path, dtype=torch.float32)

        if no_forward_exit:
            coords_list.append(path_tensor[: -1,:])
        else:
            coords_list.append(path_tensor)

    coords = torch.cat(coords_list, dim=0)  # (sum(K_i), 2)

    #model.eval()
    with torch.no_grad():
        outputs = model(coords).reshape(-1)  # (sum(K_i),)


    x = coords[:, 0].cpu().numpy()
    y = coords[:, 1].cpu().numpy()
    c = outputs.cpu().numpy()
    c_max = torch.max(outputs).cpu().numpy()
    c_min = torch.min(outputs).cpu().numpy()

    plt.figure(figsize=figsize)

    # --- Exit-Points ---
    for item in [exits_exact, exits]:

        if item is not None and len(item) > 0:
            exit_list = []
            for ep in item:
                if isinstance(ep, torch.Tensor):
                    ep_tensor = ep.clone().detach().float()
                else:
                    ep_tensor = torch.tensor(ep, dtype=torch.float32)
                exit_list.append(ep_tensor)
            exit_coords = torch.stack(exit_list, dim=0)  # (M, 2)

            # Modell-Output für Exit-Points
            with torch.no_grad():
                exit_outputs = model(exit_coords).reshape(-1)


            c_max = max(c_max, torch.max(exit_outputs).cpu().numpy())
            c_min = min(c_min, torch.min(exit_outputs).cpu().numpy())
            norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)

            plt.scatter(exit_coords[:,0].cpu().numpy(),
                        exit_coords[:,1].cpu().numpy(),
                        c=exit_outputs.cpu().numpy(),
                        cmap=cmap,
                        norm=norm,
                        s=5,
                        alpha=0.7
                        )
    # Color Map needs if condition to run before scatter for norm
    norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
    scatter = plt.scatter(x, y, c=c, cmap=cmap, norm=norm, s=7, alpha=0.7)

    plt.colorbar(scatter, label="Model Output")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    #plt.show()
    #plt.draw()
    #plt.pause(0.1)
    if no_forward_exit:
        plt.savefig('output/plots/interior_training_%d' % iter)
    else:
        plt.savefig('output/plots/training_%d'%iter)
    plt.close()
    pass


def format_samples(paths, cut_forward_exit=False):
    dimensions = paths[0].shape[0]
    coords_list = []
    for path in paths:
        if isinstance(path, torch.Tensor):
            path_tensor = path.clone().float()
        else:
            path_tensor = torch.tensor(path, dtype=torch.float32)
        if cut_forward_exit:
            coords_list.append(path_tensor[: -1, :])
        else:
            coords_list.append(path_tensor)
    if dimensions > 2:
        return torch.cat(coords_list, dim=0)
    else:
        return torch.stack(coords_list, dim=0)


def to_numpy(t):
    t = t.cpu()
    if t.requires_grad:
        t = t.detach()
    return t.numpy()


def plot_model(coords, vals=None, save_as="test_plot", figsize=(6, 6), cmap="coolwarm", relative_plot=False, title="Plot", vals_label="u(x,y)", iter=0, no_forward_exit=False):
    fig, ax = plt.subplots()
    if vals is not None:
        x = to_numpy(coords[:, 0])
        y = to_numpy(coords[:, 1])
        c = to_numpy(vals)

        c_max = torch.max(vals).cpu().detach().numpy()
        c_min = torch.min(vals).cpu().detach().numpy()

        if relative_plot:
            #cmap = 'viridis'
            norm = mpl.colors.LogNorm(vmin=max(c_min, 1e-10), vmax=c_max)
        else:
            norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)

        scatter = plt.scatter(x, y, c=c, cmap=cmap, norm=norm, s=5, alpha=0.7)
        if vals_label is not None:
            plt.colorbar(scatter, label=vals_label)
        else:
            plt.colorbar(scatter)
    else:
        # plot trajectories
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for i, path in enumerate(coords):
            x = to_numpy(path[:, 0])
            y = to_numpy(path[:, 1])

            plt.scatter(x, y, color=colors[i], s=15, alpha=1.0)
            plt.plot(x, y, color=colors[i], linewidth=1.0, alpha=0.9)
        # create circle
        circle = mpl.patches.Circle((0.5, 0.5), 0.5, facecolor='lightgray', edgecolor=None, linewidth=0, zorder=0)
        ax.add_patch(circle)

    if vals_label is not None:
        plt.xlabel("x")
        plt.ylabel("y")
    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.axis("equal")
    plt.savefig(save_as)
    plt.close()
    pass


def plot_timelines(model_vals, vals_exact=None, save_as="timelines", stepsize=0.01):
    plt.figure()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    light_colors = [
        "#92c5de",  # light blue
        "#fbb77f",  # light orange
        "#a6dba0",  # light green
        "#f4a6a6",  # light red
        "#d4b4e8",  # light purple
    ]

    for i, vals in enumerate(model_vals):
        c = to_numpy(vals)
        plt.plot(c, color=colors[i], label="model" if i == 0 else None, linewidth=0.8)

    if vals_exact is not None:
        for i, vals in enumerate(vals_exact):
            c = to_numpy(vals)
            plt.plot(c, color=light_colors[i], label="exact" if i == 0 else None, linestyle='--', linewidth=0.8)

    ax = plt.gca()
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick * stepsize:.3f}" for tick in ticks])

    # plot exit level -- 0 Dirichlet boundary
    plt.axhline(y=0.0, linestyle='--', color='lightgrey', linewidth=0.8, zorder=0)

    plt.xlim(0, None)
    plt.tight_layout()
    # plt.xlabel("t")
    plt.legend()
    plt.savefig(save_as)
    plt.close()
    pass


def plot_history(filename="history.txt", save_as="history_plot.png"):
    values = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                try:
                    values.append(float(line))
                except ValueError:
                    print(f"Skipping invalid line: {line}")

    if not values:
        print("history data error")
        return

    # Create plot
    plt.figure()
    plt.plot(values)

    plt.yscale("log")
    plt.grid(True)

    plt.savefig(save_as)
    plt.close()
    pass


# -------------------------------------------------
#   this should be a forward class method
def pad_forward_paths(torch_paths, forward):
    """
        list_of_series: list of tensors with (different) shapes (Ki, 2)
        Returns:
            padded: (Kmax, B, 2)
            mask:   (Kmax, B)
    """
    B = len(torch_paths)
    lengths = [s.size(0) for s in torch_paths]
    Kmax = max(lengths)

    padded = torch.zeros(Kmax, B, 2)
    mask = torch.zeros(Kmax, B)

    for b, series in enumerate(torch_paths):
        K = series.size(0)
        padded[:K, b] = series
        mask[:K, b] = 1.0

    forward.num_true = int(mask.sum().item())
    forward.num_interior = forward.num_true - forward.batch_size
    forward.path_steps = mask.sum(axis=0)
    forward.mean_steps = int(torch.mean(forward.path_steps).item())
    forward.max_steps = int(torch.max(forward.path_steps).item())
    forward.samples_boundary_exact = forward.mean_steps * forward.batch_size // 4 # // 2
    return padded, mask


class ForwardProcess:
    def __init__(self, stepsize_h=0.01, batch_size=100):
        self.stepsize_h = stepsize_h

        #self.new_batch = -7 # TrueFalse    # TODO - mmake active class attribute
        self.new_batch_iter = 500

        self.batch_size = batch_size  # 100 #450 # batch_size -- number of paths for interior

        self.samples_boundary = 10  # batch_size -- number of additional exit point via Euler Maruyama
        self.samples_boundary_exact = self.batch_size * 200 // 2  # exact exit points -- !! this is just init !! -- gets set in pad_forward_paths depending on the sampled trajectories

        self.radius = 0.5
        self.center = (0.5, 0.5)
        self.f = 0.5

        # sample parameters -- simulation result specific; filled in pad_forward_paths
        self.num_true = 0
        self.num_interior = 0
        self.path_steps = []
        self.mean_steps = 0
        self.max_steps = 0

    def exact_solution(self, points):
        c1, c2 = self.center
        vals = self.f * (self.radius**2 - (points[:,0] - c1)**2 - (points[:,1] - c2)**2)  # / 2  # MW here we solve the laplace equation. To avoid scaling of Brownian motion. Torch approximates 'PDE *1/2'
        return vals


if __name__ == "__main__":

    forward = ForwardProcess()
    model = TorchNet(in_dim=2, stepsize=forward.stepsize_h, width=20, depth=2)

    train = False
    if train:
        print('Start Training')
        #   quick training
        trained_model = train_adam(model, forward, new_batch=True, lr_adam=1e-2, epochs_adam=[2000, 15000, 5000, 5000, 5000])
        #   long training
        #trained_model = train_adam(model, forward, new_batch=True, lr_adam=1e-2, epochs_adam=[4000, 100000, 100000, 50000, 50000])
        save_model_state(trained_model, "output/net_adam_mini_batch.pth")
        print('Training comlete; model saved')
    else:
        print('No training; just evaluation.')

    # -------------------------------------------------
    #   Load a Model -- currently: load the one that was trained above!
    # model_load = load_model_state(TorchNet, filepath="output/net_adam_mini_batch.pth", stepsize=forward.stepsize_h, width=20, depth=2)

    # diss data load
    model_load = load_model_state(TorchNet, filepath="diss_used/output_torch_final_continued/sinnet_adam_new_patth.pth", stepsize=forward.stepsize_h, width=20, depth=2)    # XXX

    # -------------------------------------------------
    #   Evaluate and Plot
    #
    #   1   |   plot model for exemplary training batch
    #
    val_trajectories, _ = simulate_brownian_exit(forward.batch_size, 0, forward.stepsize_h, forward.radius, forward.center)
    val_padded, val_mask = pad_forward_paths(val_trajectories, forward)
    print(f"[Validation Data 1] |  Batch Size: {forward.batch_size}   |   Steps Total: {forward.num_true}, Mean: {forward.mean_steps}, Max: {forward.max_steps}, Interior: {forward.num_interior}  |  Exact Boundary Points: {forward.samples_boundary_exact}")
    val_exact_exits = simulate_uniform_exit_points(forward.samples_boundary_exact, forward.radius, forward.center)

    # format samples -- point cloud; no path structure
    interior_unstructured_coords = format_samples(val_trajectories, cut_forward_exit=True)
    exact_unstructured_coords = format_samples(val_exact_exits)
    unstructured_coords = torch.cat([interior_unstructured_coords, exact_unstructured_coords], dim=0)
    unstructured_coords = unstructured_coords.clone().detach().requires_grad_(True)

    # eval model -- scaling because we learned u/2
    model_vals = 2.0 * model_load(unstructured_coords).reshape(-1)
    plot_model(coords=unstructured_coords, vals=model_vals, save_as='output/load_trained_model', title=None, vals_label=None)

    #
    #   2   |   plot difference to exact solution on domain - fine batch
    #
    stepsize_fine = 0.0005
    batch_size_fine = 100
    forward = ForwardProcess(stepsize_h=stepsize_fine, batch_size=batch_size_fine)

    # generate validation data
    val_trajectories, _ = simulate_brownian_exit(forward.batch_size, 10, forward.stepsize_h, forward.radius, forward.center)
    val_padded, val_mask = pad_forward_paths(val_trajectories, forward)
    print(f"[Validation Data 2] |  Batch Size: {forward.batch_size}   |   Steps Total: {forward.num_true}, Mean: {forward.mean_steps}, Max: {forward.max_steps}, Interior: {forward.num_interior}  |  Exact Boundary Points: {forward.samples_boundary_exact}")
    val_exact_exits = simulate_uniform_exit_points(forward.samples_boundary_exact, forward.radius, forward.center)

    interior_unstructured_coords = format_samples(val_trajectories, cut_forward_exit=True)
    exact_unstructured_coords = format_samples(val_exact_exits)
    unstructured_coords = torch.cat([interior_unstructured_coords, exact_unstructured_coords], dim=0)
    unstructured_coords = unstructured_coords.clone().detach().requires_grad_(True)

    # eval exact solution
    exact_vals = forward.exact_solution(unstructured_coords).reshape(-1)
    exact_grads = gradients(exact_vals, unstructured_coords)
    exact_grad_norms = torch.norm(exact_grads, dim=-1)
    # eval model
    model_vals = 2.0 * model_load(unstructured_coords).reshape(-1)
    model_grads = gradients(model_vals, unstructured_coords)
    model_grad_norms = torch.norm(model_grads, dim=-1)
    # compute differences
    difference_vals = torch.abs(model_vals - exact_vals)    # absolute error
    difference_grads = model_grads - exact_grads
    difference_grads_norms = torch.norm(difference_grads, dim=-1)

    plot_model(coords=unstructured_coords, vals=exact_vals, save_as='output/exact_solution')
    plot_model(coords=unstructured_coords, vals=exact_grad_norms, save_as='output/exact_solution_grads')

    plot_model(coords=unstructured_coords, vals=model_grad_norms, save_as='output/load_trained_model_grads')

    # plot absolute errors
    plot_model(coords=unstructured_coords, vals=difference_vals, save_as='output/load_difference', title=None, vals_label=None)
    plot_model(coords=unstructured_coords, vals=difference_grads_norms, save_as='output/load_difference_grads', title=None, vals_label=None)

    # compute relative error for gradients
    difference_grads_norms_relative = difference_grads_norms / exact_grad_norms  # consider all values --> leads to division by zero
    diff_mask = torch.isfinite(difference_grads_norms_relative)

    plot_model(coords=unstructured_coords[diff_mask], vals=difference_grads_norms_relative[diff_mask], relative_plot=True, save_as='output/error_grads_relative', title=None, vals_label=None)

    #
    #   3   |   plot trajectories - fine batch
    #
    paths = val_trajectories[0:3]
    plot_model(paths, save_as='output/load_timelines_paths', title=None, vals_label=None)

    model_path_vals_list = [2 * model_load(path) for path in paths] #   +2 as above
    exact_path_vals_list = [forward.exact_solution(path) for path in paths]
    plot_timelines(model_vals=model_path_vals_list, vals_exact=exact_path_vals_list, stepsize=forward.stepsize_h, save_as='output/load_timelines')

    plot_history(filename="diss_used/output_torch_final/history.txt", save_as="output/loss_history")
