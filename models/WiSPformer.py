import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN
from layers.WienerishFilter import WienerishFilter
from layers.SpectralTransformer import SpectralTransformerD as SpectralTransformer


class GatedFusion(nn.Module):
    def __init__(self, 
                seq_len: int, 
                pred_len: int,
                d_ff: int = 512,
                embed_size: int = 16,
                dropout: float = 0.1,
                ):
        super().__init__()

        self.gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.seq_len = seq_len

        self.fc_y = nn.Sequential(
            nn.Linear(seq_len * embed_size, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, pred_len)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_f: torch.Tensor, x_a: torch.Tensor):
        if x_f is not None and x_a is not None:
            g = torch.sigmoid(self.gate_logit)
            delta = x_a - x_f
            x_out = x_f + g * delta
        else:
            if x_f is not None:
                x_out = x_f
            elif x_a is not None:
                x_out = x_a
        
        x_out = self.dropout(x_out)
        y_hat = self.fc_y(x_out)

        return y_hat


class Model(nn.Module):
    def init_models(self, configs):
        self.complex_filter = WienerishFilter(
            configs, 
            spike_aware=self.spike_aware, 
            base_floor_init=self.base_floor_init, 
            )

        self.fft_transformer = SpectralTransformer(configs) 
        
        self.dyn_gate = GatedFusion(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                dropout=configs.dropout,
                d_ff=self.d_ff,
                embed_size=self.embed_size,
                )  

        self.revin_layer = RevIN(self.enc_in, affine=True)


    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model  # hidden_size
        self.d_ff = configs.d_ff  # d_ff

        self.data_name = configs.data_path.split(".")[0]

        self.embed_size = configs.embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.seq_in_fre_points = int((self.seq_len + 1) / 2 + 0.5)

        self.spike_aware = "G2"
        self.base_floor_init = -4.0
        self.init_models(configs=configs)


    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2)

        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        return x * embeddings

    def forward_filter(self, x, do_irfft=True):
        B, T, N = x.shape

        x = self.tokenEmb(x, self.embeddings)
        x_filter, Z = self.complex_filter(x)

        if do_irfft:
            x_filter = torch.fft.irfft(x_filter, n=T, dim=-1, norm='ortho')
            x_filter = x_filter.flatten(-2)

        return x_filter, Z

    def forward_transformer(self, x, x_filter=None, do_irfft=True):
        B, T, N = x.shape

        x = self.tokenEmb(x, self.embeddings)
        x_attns = self.fft_transformer(x, x_filter=x_filter)

        assert len(x_attns.shape) == 4
        if do_irfft:
            x_attns = torch.fft.irfft(x_attns, n=T, dim=-1, norm='ortho')
            x_attns = x_attns.flatten(-2)
        
        return x_attns

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        x_norm = self.revin_layer(x, mode='norm')
        x_filter, _ = self.forward_filter(x_norm, do_irfft=False)

        x_hat_fft = self.forward_transformer(x_norm, x_filter=x_filter, do_irfft=True)
        out = self.dyn_gate(None, x_hat_fft).transpose(-1, -2)

        out = self.revin_layer(out, mode='denorm')

        return out