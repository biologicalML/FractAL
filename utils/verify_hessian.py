import torch
from torch.autograd import grad

def hvp(loss, params, v):
    """
    Hessian-vector product: H v
    loss: scalar loss
    params: list of parameters
    v: list of tensors with same shapes as params
    """
    grads = grad(loss, params, create_graph=True)
    hv = grad(grads, params, grad_outputs=v, retain_graph=True)
    return hv

def dot(xs, ys):
    return sum(torch.sum(x * y) for x, y in zip(xs, ys))

def norm(xs):
    return torch.sqrt(dot(xs, xs))

def add(xs, ys):
    return [x + y for x, y in zip(xs, ys)]

def sub(xs, ys):
    return [x - y for x, y in zip(xs, ys)]

def scale(xs, a):
    return [a * x for x in xs]

def check_ihvp_correct(logger, loss, params, ihvp, v, damping, tol=1e-2):
    """
    Check if the inverse Hessian-vector product is correct.
    loss: scalar loss
    params: list of parameters
    ihvp: inverse Hessian-vector product
    tol: tolerance for checking correctness
    """
    # Compute Hessian-vector product using the computed ihvp
    with torch.no_grad():
        hvp_result = hvp(loss, params, ihvp)
    norm_factor = sum(len(x.view(-1)) for x in params)
    residual = norm(sub(add(hvp_result, scale(ihvp, damping)), v))/(norm_factor) ## To control for the damping factor in the check
    print("Residual ||Hu - v|| =", residual.item())
    logger.write(f"Residual ||Hu - v|| = {residual.item()}")
    # Check if the result is close to the identity vector (i.e., v)
    return residual.item() < tol