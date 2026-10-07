import torch
from typing import Union, Sequence

from .verify_hessian import check_ihvp_correct

def flatten(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors])

def unflatten(vec, like_tensors):
    outputs = []
    idx = 0
    for t in like_tensors:
        numel = t.numel()
        outputs.append(vec[idx:idx+numel].view_as(t))
        idx += numel
    return outputs


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def hessian_vector_product(ys: torch.Tensor,
                           xs: Union[torch.Tensor, Sequence[torch.Tensor]],
                           v: Union[torch.Tensor, Sequence[torch.Tensor]]):
    """
    Multiply the Hessian of `ys` wrt `xs` by `v` using PyTorch autograd.

    Args:
        ys: scalar tensor (or a tensor that reduces to a scalar).
        xs: tensor or list/tuple of tensors to compute Hessian over.
        v: list/tuple of tensors (same shapes as xs) or a tensor (if xs single).

    Returns:
        list of tensors (or single tensor if len(xs) == 1) containing Hessian* v.
    """
    xs_list = _as_list(xs)
    v_list = _as_list(v)
    
    if len(xs_list) != len(v_list):
        raise ValueError("xs and v must have the same length.")

    # First-order gradients (create_graph=True so we can differentiate them)
    grads = torch.autograd.grad(ys, xs_list, create_graph=True, allow_unused=True)
    # grads is a tuple, possibly containing None entries
    
    # Multiply elementwise with v (detached) and sum to get a scalar
    flat_grads = [g.reshape(-1) for g in grads if g is not None]
    flat_vs = [vv.detach().reshape(-1) for g, vv in zip(grads, v_list) if g is not None]

    elemwise_sum = torch.dot(torch.cat(flat_grads), torch.cat(flat_vs))

    # Second derivatives: gradient of the scalar elemwise_sum wrt xs
    hvp = torch.autograd.grad(elemwise_sum, xs_list, allow_unused=True)
    # Replace None with zeros_like
    result = [torch.zeros_like(x) if g is None else g for x, g in zip(xs_list, hvp)]
    
    if len(result) == 1:
        return result[0]
    return result

def make_matvec(trainer, train_idxs, params, damping):
    def matvec(v_list):
        hvp = hessian_vector_product(
            ys=trainer.get_total_loss(train_idxs),
            xs=params,
            v=v_list
        )
        return [h + damping * v for h, v in zip(hvp, v_list)]
    return matvec

def make_last_layer_matvec(trainer, train_idxs, params, damping):
    """Fast path for last_layer=True: the backbone is frozen while solving for
    last_layer params, so its output (repr) is identical on every CG iteration.
    Compute it once here instead of re-running the full backbone forward pass
    (plus rebuilding a DataLoader) on every one of the CG iterations."""
    repr_, y = trainer.get_repr_and_labels(train_idxs)
    is_bio = trainer.args.dataset.startswith("bio")

    def loss_fn():
        y_pred = trainer.model.last_layer(repr_)
        if is_bio:
            loss = trainer.loss_fn(y_pred.squeeze(dim=-1), y).mean(dim=-1).sum()
        else:
            loss = trainer.loss_fn(y_pred.squeeze(dim=-1), y).sum()
        return loss / len(train_idxs)

    def matvec(v_list):
        hvp = hessian_vector_product(
            ys=loss_fn(),
            xs=params,
            v=v_list
        )
        return [h + damping * v for h, v in zip(hvp, v_list)]
    return matvec

def conjugate_gradient(logger, matvec, v, params, max_iter=100, tol=1e-8):
    """
    Solve (H + λI) u = v using CG
    """
    v_flat = flatten(v)
    u = torch.zeros_like(v_flat)

    r = v_flat.clone()          # r0 = v - A u0, u0 = 0
    p = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(max_iter):
        p_list = unflatten(p, params)
        Ap_list = matvec(p_list)
        Ap = flatten(Ap_list)
        # print(Ap.device)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        u = u + alpha * p
        r = r - alpha * Ap

        rs_new = torch.dot(r, r)
        if i % 20 == 0:
            print(f"CG iteration {i}, residual norm: {torch.sqrt(rs_new).item()}")
        if i % 100 == 0 or i == max_iter - 1:
            logger.write(f"CG iteration {i}, residual norm: {torch.sqrt(rs_new).item()}")
        if torch.sqrt(rs_new) < tol:
            print(f"CG converged in {i+1} iterations.")
            logger.write(f"CG converged in {i+1} iterations.")
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return unflatten(u, params)

def get_inverse_hvp_cg(
    logger,
    trainer,
    v,
    train_idxs,
    damping=0.001,
    cg_iters=100,
    tol=1e-8,
    verify_tol = 1e-2,
    last_layer=False
):
    params = trainer.get_all_params(last_layer=last_layer)
    v = [vi.detach() for vi in v]

    matvec_factory = make_last_layer_matvec if last_layer else make_matvec
    matvec = matvec_factory(
        trainer=trainer,
        train_idxs=train_idxs,
        params=params,
        damping=damping
    )

    ihvp = conjugate_gradient(
        logger,
        matvec=matvec,
        v=v,
        params=params,
        max_iter=cg_iters,
        tol=tol
    )
    return ihvp
