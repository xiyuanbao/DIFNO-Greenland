"""Gradient surgery for the two-task (data loss + cycle loss) inverse training:
PCGrad projection (Yu et al. 2020) used by the default inverse, and CAGrad
(Liu et al. 2021, c = 0.2) used by the coastal inverse. Both take the two
per-parameter gradient lists and return one combined gradient list.
"""

import torch
from scipy.optimize import minimize_scalar


def pcgrad_two_task(grads_inv, grads_cycle, lambda_cycle):
    """PCGrad: drop the component of the cycle gradient that conflicts with the
    data gradient, then add it back scaled by lambda_cycle.

    Extracted verbatim from the training loop of
    ``no_smooth_512full_cycle_fingerprint_FNO_dse.ipynb``.
    """
    projected_grads = []
    for g1, g2 in zip(grads_inv, grads_cycle):
        dot = torch.dot(g1.flatten(), g2.flatten()).real
        if dot < 0:
            g2_proj = g2 - (dot / (g1.norm()**2 + 1e-6)) * g1
        else:
            g2_proj = g2
        projected_grads.append(g1 + lambda_cycle * g2_proj)
    return projected_grads


def _mgda_two_task(g1_flat, g2_flat, eps=1e-12):
    # Solve: min_{w in [0,1]} || w*g1 + (1-w)*g2 ||^2
    # Closed-form for 2-task MGDA
    d = g1_flat - g2_flat
    denom = torch.vdot(d, d).real + eps
    w = torch.vdot(d, g1_flat).real / denom
    w = torch.clamp(w, 0.0, 1.0)
    return w


def _cagrad_two_task(grads1, grads2, c=0.2, eps=1e-12):
    g0 = [0.5 * (a + b) for a, b in zip(grads1, grads2)]

    g0_norm = torch.norm(torch.stack([g.norm() for g in g0])).real.item()
    coef = float(c) * g0_norm

    def obj(x):
        x_t = torch.tensor(float(x), device=g0[0].device, dtype=g0[0].dtype)
        gw = [x_t * a + (1.0 - x_t) * b for a, b in zip(grads1, grads2)]
        gw_norm = torch.norm(torch.stack([g.norm() for g in gw])).real

        # g_w^T g_0 (use vdot for complex stability), return scalar for scipy
        dot = torch.tensor(0.0, device=g0[0].device, dtype=torch.float32)
        for a, b in zip(gw, g0):
            dot = dot + torch.vdot(a.reshape(-1), b.reshape(-1)).real.to(torch.float32)

        return (coef * gw_norm.to(torch.float32) + dot).item()

    res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
    x = float(res.x)
    x_t = torch.tensor(x, device=g0[0].device, dtype=g0[0].dtype)

    gw = [x_t * a + (1.0 - x_t) * b for a, b in zip(grads1, grads2)]
    gw_norm = torch.norm(torch.stack([g.norm() for g in gw])).real

    lmbda = coef / (gw_norm.item() + 1e-4)
    g = [g0_i + lmbda * gw_i for g0_i, gw_i in zip(g0, gw)]
    g = [gi / (1.0 + float(c)) for gi in g]

    return g


def _match_param_dtype(grads, params):
    # Ensure each grad matches the corresponding parameter dtype (and is real if param is real)
    out = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p))
            continue
        if torch.is_complex(g) and (not torch.is_complex(p)):
            g = g.real
        out.append(g.to(dtype=p.dtype))
    return out
