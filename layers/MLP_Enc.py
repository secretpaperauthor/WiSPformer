import torch


class MLPEnc(torch.nn.Module):
    def __init__(self, num: int, d_model: int, blocks: int, dropout=0.1, use_series_wise=True):
        super().__init__()
        self.blocks = blocks
        self.use_series_wise = use_series_wise

        self.mlp = torch.nn.ModuleList()
        self.norm1 = torch.nn.ModuleList()
        if use_series_wise:
            self.mlp_s = torch.nn.ModuleList()
            self.norm2 = torch.nn.ModuleList()

        for i in range(self.blocks):
            self.mlp.append(
                torch.nn.Sequential(
                    torch.nn.Linear(d_model, d_model),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout)
                )
            )
            self.norm1.append(torch.nn.LayerNorm(d_model))
            if use_series_wise:
                self.mlp_s.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(num, num),
                        torch.nn.GELU(),
                        torch.nn.Dropout(dropout)
                    )
                )
                self.norm2.append(torch.nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.blocks):
            x = x + self.mlp[i](x)  # [B,num,d_model]
            x = self.norm1[i](x)
            if self.use_series_wise:
                x = x + self.mlp_s[i](x.permute(0, 2, 1)).permute(0, 2, 1)
                x = self.norm2[i](x)

        return x