import math
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as tF

from layers.RevIN import RevIN
from models.CFBoost import MultiChannelSpectralFilter


class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(GCNLayer, self).__init__()
        self.lin = nn.Linear(c_in, c_out)

    def forward(self, H, A):  # H:(B,N,C), A:(B, N,N) row-stochastic
        Hn = torch.einsum("bij,bjc->bic", A, H)  # A @ H
        # Hn = torch.einsum("ij,bjc->bic", A, H)  # A @ H
        return tF.relu(self.lin(Hn))


class ComplexLinear(nn.Module):
    def __init__(self, d_in, d_out, scale=0.02):
        super(ComplexLinear, self).__init__()
        self.Wr = nn.Parameter(torch.randn(d_in, d_out) * scale)
        self.Wi = nn.Parameter(torch.randn(d_in, d_out) * scale)
        self.br = nn.Parameter(torch.zeros(d_out))
        self.bi = nn.Parameter(torch.zeros(d_out))

    def forward(self, z):  # z: (..., d_in) complex
        xr, xi = z.real, z.imag
        yr = xr @ self.Wr - xi @ self.Wi + self.br
        yi = xi @ self.Wr + xr @ self.Wi + self.bi
        return torch.complex(yr, yi)


def modrelu(z, bias, eps=1e-6):
    # bias: real, broadcastable 到最后一维
    r = torch.abs(z)
    scale = torch.relu(r + bias) / (r + eps)
    return z * scale


# dimension extension
def tokenEmb(self, x, embeddings):
    if self.embed_size <= 1:
        return x.transpose(-1, -2).unsqueeze(-1)
    # x: [B, T, N] --> [B, N, T]
    x = x.transpose(-1, -2)
    x = x.unsqueeze(-1)
    # B*N*T*1 x 1*D = B*N*T*D
    return x * embeddings


class MTS_FFTGNN(nn.Module):
    def __init__(self, configs) -> None:
        super(MTS_FFTGNN, self).__init__()

        self.pred_len = configs.pred_len  # H
        self.enc_in = configs.enc_in  # N sensor channels
        self.seq_len = configs.seq_len # look back window T
        self.embed_size = configs.embed_size
        self.d_ff = configs.d_ff
        self.use_coh_freq_attn = configs.coh_freq_attn
        self.topk = configs.topk
        
        self.F_in_dim = int((self.seq_len + 1) / 2 + 0.5)

        self.n_gnn_layers = configs.gnn_layers
        self.n_gnn_hidden = configs.gnn_hidden
        gnn_layers = []
        if self.n_gnn_layers > 0:
            for i in range(self.n_gnn_layers):
                if i == 0:
                    gnn_layers.append(GCNLayer(self.F_in_dim * self.embed_size, self.gnn_hidden))
                elif i == self.n_gnn_layers - 1:
                    gnn_layers.append(GCNLayer(self.gnn_hidden, self.F_in_dim * self.embed_size))
        self.gnn_module = nn.ModuleList(gnn_layers)

        self.decode_fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        if self.use_coh_freq_attn:
            self.coh_freq_logits = nn.Parameter(torch.zeros(self.F_in_dim))

        if self.use_filter:
            self.complex_filter = MultiChannelSpectralFilter(configs, use_single=True)

        self.w_z = ComplexLinear(self.F_in_dim, self.F_in_dim)

        self.log_tau = nn.Parameter(torch.tensor(math.log(1e-3)))

        self.revin_layer = RevIN(self.enc_in, affine=True)


    def build_graph(self, x_emb):
        B, N, T, D = x_emb.shape

        # [B, N, D, T]
        x_emb = x_emb.transpose(-1, -2)

        # fft
        x_fre = torch.fft.rfft(x_emb, dim=-1, norm='ortho')  # FFT on L dimension

        # [B, N, D, fre_points]
        assert x_fre.shape[-1] == self.F_in_dim

        B, N, D, F = x_fre.shape

        x_fre_ = self.w_z(x_fre)

        # Sxx shape [N, D, F]
        Sxx = (x_fre_.abs() ** 2).mean(dim=(0,2)) # [N,F]
        Sxy = (x_fre_[:, :, None, :] * x_fre_[:, None, :, :].conj()).mean(dim=(0,3))  # (N,N,F)
        Coh = (torch.abs(Sxy) ** 2) / (
            Sxx[:, None, :] * Sxx[None, :, :] + 1e-6
        )  # (N,N,F)

        if self.use_coh_freq_attn:
            wf = torch.softmax(self.coh_freq_logits[:F], dim=0)  # (F,)
            A = (Coh * wf[None, None, :]).sum(-1)  # (N,N)
        else:
            A = Coh.mean(-1)
        
        # 稀疏化
        if self.topk is not None:
            k = min(int(self.topk), self.N)
            idx = torch.topk(A, k, dim=-1).indices
            mask = torch.zeros_like(A).scatter_(-1, idx, 1.0)
            A = A * mask

            tau = torch.sigmoid(self.log_tau)
            A = torch.softmax(A / tau, dim=-1)

        return A

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, T, N = x.shape

        # revin norm
        x_norm = self.revin_layer(x, mode='norm')
        # x_a [B, N, T, D]
        x_a = self.tokenEmb(x_norm, self.embeddings)

        A = self.build_graph(x_a)

        if self.use_filter:
            # z_filter [B, N, D, F] 
            z_filter, z_ori = self.complex_filter(x_a)
            z_gnn = self.gnn_module(z_filter.flatten(-2))
            
