"""DIFNO model classes, taken from the training notebooks with one change:
every hard-coded ``.cuda()`` becomes ``.to(device)`` so the code also runs on
MPS / CPU. ``VFT`` is the Vandermonde (direct spectral evaluation) transform,
``SpectralConv2d_dse`` one Fourier layer, ``FNO_dse`` the tied-weight operator
that iterates that single layer and projects onto a different output point set.

Device note: the complex products in the spectral layer and in the Vandermonde
transform are routed through :func:`complex_einsum`, which on Apple MPS
evaluates the complex contraction as four real einsums and materialises
conjugated / permuted complex views. This works around PyTorch MPS bugs in the
complex-einsum backward and in complex kernels fed conj views, so MPS results
match CUDA / CPU up to float precision. On CUDA and CPU the original code path
is used unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default device used when a VFT / FNO_dse is built without an explicit one.
_DEFAULT_DEVICE = torch.device("cpu")


def set_default_device(device):
    """Set the device used by VFT / FNO_dse when configs has no 'device' key."""
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = torch.device(device)


# ----------------------------------------------------------------------------
# MPS-safe complex helpers (CUDA / CPU paths unchanged)
# ----------------------------------------------------------------------------
# PyTorch's MPS backend has two bugs that affect this model:
#  1. the complex einsum / bmm BACKWARD drops the conjugation of the saved
#     operand, so gradients of the spectral weights are wrong on MPS;
#  2. complex kernels silently return wrong values for operands that carry a
#     lazy conj bit and are non-contiguous (the VFT inverse matrix, built as
#     conj(V).permute(0, 2, 1), is exactly that).
# Real-valued einsum is correct on MPS, so on MPS we expand
# (a + bi)(c + di) = (ac - bd) + i(ad + bc) into four real einsums and
# materialise conj views. Both helpers are exact no-ops in effect off MPS.


def _mps_complex_einsum(eq, A, B):
    Ar, Ai = A.real.contiguous(), A.imag.contiguous()
    Br, Bi = B.real.contiguous(), B.imag.contiguous()
    re = torch.einsum(eq, Ar, Br) - torch.einsum(eq, Ai, Bi)
    im = torch.einsum(eq, Ar, Bi) + torch.einsum(eq, Ai, Br)
    return torch.complex(re, im)


def complex_einsum(eq, A, B):
    """torch.einsum for complex operands, real-decomposed on Apple MPS."""
    if A.device.type == "mps" or B.device.type == "mps":
        return _mps_complex_einsum(eq, A, B)
    return torch.einsum(eq, A, B)


def _materialize_complex(t):
    """Resolve a lazy conj bit and make contiguous, on MPS only."""
    if t.device.type == "mps" and t.is_complex():
        return t.resolve_conj().contiguous()
    return t


# class for fully nonequispaced 2d points
class VFT_changing_points:
    def __init__(self, x_positions, y_positions, modes, device=None):
        device = torch.device(device) if device is not None else _DEFAULT_DEVICE
        self.device = device
        # it is important that positions are scaled between 0 and 2*pi
        # the xy data here is already scaled
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().to(device)
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes-1), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().to(device)

        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes*2)*(self.modes*2-1)
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.modes*2-1), 1)
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.modes*2).reshape(self.batch_size,m,self.number_points))
        forward_mat = torch.exp(-1j* (X_mat+Y_mat)) / np.sqrt(self.number_points) * np.sqrt(2)

        inverse_mat = _materialize_complex(torch.conj(forward_mat.clone()).permute(0,2,1))

        return forward_mat, inverse_mat

    def forward(self, data):
        if data.device.type == "mps":
            return complex_einsum('bmn,bnc->bmc', self.V_fwd, data)
        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        if data.device.type == "mps":
            return complex_einsum('bnm,bmc->bnc', self.V_inv, data)
        data_inv = torch.bmm(self.V_inv, data)

        return data_inv


class VFT:#fixed points
    def __init__(self, x_positions, y_positions, modes, device=None):
        device = torch.device(device) if device is not None else _DEFAULT_DEVICE
        self.device = device
        # Positions are shared across all batches, shape: (N,)
        self.x_positions = x_positions[None, :].float().to(device)  # -> shape (1, N)
        self.y_positions = y_positions[None, :].float().to(device)
        self.number_points = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(-modes, 0)), 0)[None, :, None].float().to(device)  # shape: (1, 2modes, 1)
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(-(modes - 1), 0)), 0)[None, :, None].float().to(device)  # shape: (1, 2modes-1, 1)

        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes * 2) * (self.modes * 2 - 1)

        X_mat = torch.bmm(self.X_, self.x_positions[:, None, :])  # (1, 2modes, N)
        X_mat = X_mat.repeat(1, self.modes * 2 - 1, 1)  # (1, m, N)

        Y_mat = torch.bmm(self.Y_, self.y_positions[:, None, :])  # (1, 2modes-1, N)
        Y_mat = Y_mat.repeat(1, 1, self.modes * 2).reshape(1, m, self.number_points)  # (1, m, N)

        forward_mat = torch.exp(-1j * (X_mat + Y_mat)) / np.sqrt(self.number_points) * np.sqrt(2)
        inverse_mat = _materialize_complex(torch.conj(forward_mat.clone()).permute(0, 2, 1))  # (1, N, m)

        return forward_mat, inverse_mat

    def forward(self, data):
        # data shape: (B, N, C)
        if data.device.type == "mps":
            # same contraction, broadcast without materializing (B, m, N)
            return complex_einsum('omn,bnc->bmc', self.V_fwd, data)
        B = data.shape[0]
        return torch.bmm(self.V_fwd.expand(B, -1, -1), data)

    def inverse(self, data):
        if data.device.type == "mps":
            return complex_einsum('onm,bmc->bnc', self.V_inv, data)
        B = data.shape[0]
        return torch.bmm(self.V_inv.expand(B, -1, -1), data)


class SpectralConv2d_dse (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, transformer_in, transformer_out):
        super(SpectralConv2d_dse, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        self.transformer_in, self.transformer_out =  transformer_in,transformer_out

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return complex_einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x = x.permute(0, 2, 1) #result [batchsize, Npoints, width]

        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = self.transformer_in.forward(x.cfloat()) #result [batchsize, modes^2, width]
        x_ft = x_ft.permute(0, 2, 1)
        x_ft = torch.reshape(x_ft, (batchsize, self.in_channels, 2*self.modes1, 2*self.modes1-1))

        # # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes1, self.modes1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes1] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes1], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes1] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes1], self.weights2)

        # #Return to physical space
        x_ft = torch.reshape(out_ft, (batchsize, self.out_channels, 2*self.modes1**2))
        x_ft2 = _materialize_complex(x_ft[..., 2*self.modes1:].flip(-1, -2).conj())
        x_ft = torch.cat([x_ft, x_ft2], dim=-1)

        x_ft = x_ft.permute(0, 2, 1)
        x = self.transformer_out.inverse(x_ft) # result [batchsize, Npoints, width]
        x = x.permute(0, 2, 1) # result [batchsize, wdith, Npoints]

        return x.real


#shared weight
class FNO_dse (nn.Module):
    """Tied-weight ("iterative") DSE-FNO used by all four trained operators.

    A single ``SpectralConv2d_dse`` (conv0) and a single pointwise conv (w0) are
    applied four times; the last application swaps the inverse transform from
    ``transform1`` (input point cloud) to ``transform2`` (output point cloud),
    which is what lets input and output live on different, mismatched domains.
    The linear bypass ``proj_w0`` maps N_in points to N_out points and carries
    the amplitude information the band-limited spectral branch discards.

    ``configs['proj_w0_sequential']`` selects how ``proj_w0`` is registered: the
    forward and default-inverse runs wrapped it in ``nn.Sequential`` (checkpoint
    keys ``proj_w0.0.*``), the CAGrad coastal run used a bare ``nn.Linear``
    (keys ``proj_w0.*``). It changes the parameter names only, not the maths.
    """

    def __init__(self, configs):
        super(FNO_dse, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        sparse_in = configs['point_data_in']
        sparse_out = configs['point_data_out']
        N_in, N_out = sparse_in.shape[0], sparse_out.shape[0]
        self.sparse_x_in, self.sparse_y_in = sparse_in[:,0],sparse_in[:,1]#!!!!!!
        self.sparse_x_out, self.sparse_y_out = sparse_out[:,0],sparse_out[:,1]

        self.padding = 2 # pad the domain if input is non-periodic

        # Predictions are normalized, we need the output denormalized
        self.denormalizer = configs['denormalizer']

        device = configs.get('device', _DEFAULT_DEVICE)
        transform1 = VFT(self.sparse_x_in, self.sparse_y_in, self.modes1, device=device)
        transform2 = VFT(self.sparse_x_out, self.sparse_y_out, self.modes1, device=device)
        self.transform1 = transform1
        self.transform2 = transform2
        self.fc0 = nn.Linear(1, self.width)#!!!!!!
        # input channel is just the sea level data

        self.conv0 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2, transform1, transform1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)

        if configs.get('proj_w0_sequential', True):
            self.proj_w0 = nn.Sequential(
                nn.Linear(N_in, N_out),
            )
        else:
            self.proj_w0 = nn.Linear(N_in, N_out)

        self.fc1 = nn.Linear(self.width, 32)
        self.fc2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x_skip = x

        self.conv0.transformer_out = self.transform1
        x1 = self.conv0(x)
        x2 = self.w0(x)  # [B, width, N_in]
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2 + x_skip
        x = F.gelu(x)

        x_skip = x

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        self.conv0.transformer_out = self.transform2
        x1 = self.conv0(x)
        x2 = self.proj_w0(x + x_skip)  # -> [B, width, N_out]
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)

        # x = self.denormalizer(x)
        return x
