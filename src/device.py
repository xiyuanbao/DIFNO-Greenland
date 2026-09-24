"""Device selection for DIFNO: cuda -> mps -> cpu, overridable via DIFNO_DEVICE.

The training notebooks hard-coded ``torch.device('cuda')``; this module keeps the
same code runnable on a Mac (MPS or CPU), which is why the model classes take a
``device`` instead of calling ``.cuda()``.
"""

import os

import torch


def _mps_ok():
    """MPS must exist and support the complex ops used by the spectral layer."""
    if not torch.backends.mps.is_available():
        return False
    try:
        x = torch.randn(1, 2, 4, 3, dtype=torch.cfloat, device="mps")
        w = torch.randn(2, 2, 2, 2, dtype=torch.cfloat, device="mps")
        o = torch.einsum("bixy,ioxy->boxy", x[:, :, :2, :2], w)
        buf = torch.zeros(1, 2, 4, 2, dtype=torch.cfloat, device="mps")
        buf[:, :, :2, :2] = o
        buf.flip(-1, -2).conj().reshape(1, 2, -1)
        return True
    except Exception:
        return False


def get_device(prefer=None):
    """cuda -> mps -> cpu. Override with the arg or the DIFNO_DEVICE env var."""
    prefer = prefer or os.environ.get("DIFNO_DEVICE")
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_ok():
        return torch.device("mps")
    return torch.device("cpu")


_VFT_TENSOR_ATTRS = ("x_positions", "y_positions", "X_", "Y_", "V_fwd", "V_inv")


def move_model(model, device):
    """Move an FNO_dse, including the raw tensors held by its VFT helpers.

    ``VFT`` is a plain class, not an ``nn.Module``, so ``model.to(device)`` does
    not touch its Vandermonde matrices; they have to be moved explicitly.
    """
    model.to(device)
    seen = set()
    candidates = [getattr(model, "transform1", None), getattr(model, "transform2", None)]
    for mod in model.modules():
        candidates.append(getattr(mod, "transformer_in", None))
        candidates.append(getattr(mod, "transformer_out", None))
    for vft in candidates:
        if vft is not None and id(vft) not in seen and hasattr(vft, "V_fwd"):
            for name in _VFT_TENSOR_ATTRS:
                t = getattr(vft, name, None)
                if torch.is_tensor(t):
                    setattr(vft, name, t.to(device))
            vft.device = torch.device(device)
            seen.add(id(vft))
    return model
