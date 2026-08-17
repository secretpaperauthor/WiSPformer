import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def filter_reshape(G, Z):
    if Z.ndim not in (3, 4):
        raise ValueError(f"Expected Z to have 3 or 4 dims, got {Z.ndim}")
    
    if Z.ndim == 4:
        G = G.unsqueeze(-2)
        
    return G


class WienerishFilter(nn.Module):
    """
    Wiener-ish multivariate spectral filter
    """
    def __init__(self, 
                 configs, 
                 spike_aware="G0", 
                 base_floor_init = -1.0,
                 eps=1e-6,
    ):
        super(WienerishFilter, self).__init__()
        print(f"==>> Init spike-aware Wiener-ish Fourier filter. [Model: {spike_aware}]")

        self.n_features = configs.enc_in
        self.T = configs.seq_len
        self.Fdim = self.T // 2 + 1
        self.d_ff = configs.d_ff
        self.spike_aware = spike_aware

        self.filt_wiener_phi = nn.Parameter(torch.zeros(self.Fdim))

        self.filter_fc = nn.Sequential(
            nn.Linear(self.Fdim, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.Fdim)
            )

        if self.spike_aware != "G0":
            self.eps = eps
            self.base_floor_init = base_floor_init
            self._init_spike_aware()

    def _init_spike_aware(self):
        self.spike_tau = nn.Parameter(torch.zeros(1, self.n_features, 1))      # threshold
        self.spike_alpha = nn.Parameter(torch.tensor(0.0))    # mix peak vs roughness

        if self.spike_aware == "G2":
            self.base_floor_raw = nn.Parameter(torch.ones(1, self.n_features, 1) * self.base_floor_init)


    def _spike_protect(self, x_time):
        if x_time.dim() == 4:
            # [B, N, D, T] -> [B, N, D*T]
            xt = x_time.flatten(-2)
        else:
            xt = x_time

        mu = xt.mean(dim=-1, keepdim=True)
        sd = xt.std(dim=-1, keepdim=True) + self.eps

        # peakiness: large local amplitude deviation
        peak = (xt - mu).abs().amax(dim=-1, keepdim=True) / sd

        # roughness: average first-difference magnitude
        diff = xt[..., 1:] - xt[..., :-1]
        rough = diff.abs().mean(dim=-1, keepdim=True) / sd

        # mix peak vs roughness
        a = torch.sigmoid(self.spike_alpha)
        score = a * peak + (1.0 - a) * rough

        # high score => protect -> 1
        protect = torch.sigmoid((score - self.spike_tau))

        return protect, (peak, rough)
    
    def _floor_filter(self, Z, x, G0):
        protect, pr = self._spike_protect(x)

        if Z.dim() != 3:
            protect = protect.unsqueeze(2)

        floor = protect + (1.0 - protect) * G0 #1
        return floor, protect, pr

    def _G_spike_protect_filter(self, Z, protect, G0):
        base_floor = torch.sigmoid(self.base_floor_raw)

        if Z.dim() != 3:
            base_floor = base_floor.unsqueeze(2)

        floor = base_floor + (1.0 - base_floor) * protect

        # Final spike-aware floor Wiener-ish gain
        G2 = floor + (1.0 - floor) * G0
        return G2

    def _make_filter(self, Z, x):
        if self.spike_aware == "G2":
            _, protect, _ = self._floor_filter(Z, x, G0)
            G = self._G_spike_protect_filter(Z, protect, G0)

        W = G.to(torch.complex64)
        return W, G

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.transpose(-1, -2)
        else:
            x = x.transpose(-1, -2).contiguous()
        
        Z = torch.fft.rfft(x, dim=-1, norm="ortho")
        W, _ = self._make_filter(Z, x)
        out_x = Z * W

        return out_x, Z
