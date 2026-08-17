import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention_ablation


class SpectralTransformerD(nn.Module):
    def __init__(self, configs) -> None:
        super(SpectralTransformerD, self).__init__()

        # N: enc_in, T: seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.embed_size = configs.embed_size 

        self.seq_in_fre_points = int((self.seq_len + 1) / 2 + 0.5)

        self.a = nn.Parameter(torch.ones(self.seq_in_fre_points))
        self.b = nn.Parameter(torch.zeros(self.seq_in_fre_points))

        print("# Use Fourier Attention SpectralTransformerD")
        self.encoder_fre_real = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(False, configs.factor, 
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention,
                                               token_num=configs.enc_in),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            one_output=True,
        )

        self.fre_trans_real = nn.Sequential(
            nn.Linear(self.seq_in_fre_points * self.embed_size * 2, self.d_model),
            self.encoder_fre_real,
            nn.Linear(self.d_model, self.seq_in_fre_points * self.embed_size)
        )

        self.encoder_fre_imag = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(False, configs.factor, 
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention,
                                               token_num=configs.enc_in),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            one_output=True,
        )

        self.fre_trans_imag = nn.Sequential(
            nn.Linear(self.seq_in_fre_points * self.embed_size * 2, self.d_model),
            self.encoder_fre_imag,
            nn.Linear(self.d_model, self.seq_in_fre_points * self.embed_size)
        )

    def forward(self, x, x_filter=None):
        B, N, T, D = x.shape
        assert T == self.seq_len

        x = x.transpose(-1, -2)

        # fft
        x_fre_raw = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT on L dimension
        assert x_fre_raw.shape[-1] == self.seq_in_fre_points

        if x_filter is not None:
            x_fre = torch.cat([x_fre_raw, x_filter], dim=-1)
        else:
            x_fre = x_fre_raw

        x_real, x_imag = x_fre.real, x_fre.imag

        x_real = self.fre_trans_real(x_real.flatten(-2)).reshape(B, N, D, self.seq_in_fre_points)
        x_imag = self.fre_trans_imag(x_imag.flatten(-2)).reshape(B, N, D, self.seq_in_fre_points)

        y_real = self.a * x_real - self.b * x_imag
        y_imag = self.a * x_imag + self.b * x_real

        out = torch.complex(y_real, y_imag) + x_fre_raw
        return out
