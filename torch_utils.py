# -------------------------------------------------
#   Author: Stephan Schmidt | Maximilian Würschmidt
#
#all stuff for differential operators in torch

import torch
import torch.autograd as autograd

def grad(u, x, retain_graph=True, create_graph=True, device=None):
    adj_inp = torch.ones(u.shape, device=device)
    r, *_ = autograd.grad(u, x, adj_inp, retain_graph=retain_graph, create_graph=create_graph)
    return r

def div(u, x, retain_graph=True, create_graph=True, device=None):
    """
        u: Shape [BATCH_SIZE, d, ...]
    """
    shape = u.shape
    dim = shape[1]
    r = 0.
    for i in range(dim):
        adj_inp = torch.zeros(shape, device=device)
        adj_inp[:, i, ...] = 1
        r_, *_ = autograd.grad(u, x, adj_inp, retain_graph=retain_graph, create_graph=create_graph)
        r += r_[:, i].unsqueeze(1)
    return r

def laplace(u, x):
    return div(grad(u, x), x)

# -------------------------------------------------
#   NEW for Probabilistic Network
# -------------------------------------------------
def gradients(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def hessian_vector_product(u, x, v):
    grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    hvp = torch.autograd.grad((grad * v).sum(), x, retain_graph=True)[0]
    return hvp
