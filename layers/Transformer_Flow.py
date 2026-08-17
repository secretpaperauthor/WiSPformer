import torch


class Permutation(torch.nn.Module):

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward_spda(
            self, x: torch.Tensor, mask: torch.Tensor, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale ** 2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
            self, x: torch.Tensor, mask: torch.Tensor, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum('bmhd,bnhd->bmnh', q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
            self, x: torch.Tensor, mask: torch.Tensor, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
            self, x: torch.Tensor, attn_mask: torch.Tensor, attn_temp: float = 1.0,
            which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
            self,
            d_input: int,
            d_model: int,
            d_cond: int,
            num_variables: int,
            permutation: Permutation,
            num_layers: int = 1,
            head_dim: int = 64,
            expansion: int = 1,
            nvp: bool = True,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(d_input, d_model)
        self.adaptor = torch.nn.Linear(d_cond, d_model)
        self.pos_embed = torch.nn.Parameter(torch.randn(num_variables, d_model) * 1e-2)
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(d_model, head_dim, expansion) for _ in range(num_layers)]
        )
        self.nvp = nvp
        output_dim = d_input * 2 if nvp else d_input
        self.norm = torch.nn.LayerNorm(d_model)
        self.proj_out = torch.nn.Linear(d_model, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_variables, num_variables)))

    def forward(self, x, cond):
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        x = self.proj_in(x) + pos_embed + self.adaptor(cond)

        for block in self.attn_blocks:
            x = block(x, self.attn_mask)
        x = self.norm(x)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])


class TarFlow(torch.nn.Module):

    def __init__(
            self,
            num_variables: int,
            d_input: int,
            d_model: int,
            d_cond: int,
            num_blocks: int,
            layers_per_block: int,
            nvp: bool = True
    ):
        super().__init__()
        self.num_variables = num_variables
        self.d_model = d_model
        permutations = [PermutationIdentity(self.num_variables), PermutationFlip(self.num_variables)]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    d_input,
                    d_model,
                    d_cond,
                    num_variables,
                    permutations[i % 2],
                    layers_per_block,
                    nvp=nvp,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x, cond):
        outputs = []
        logdets = torch.zeros((), device=x.device)
        for block in self.blocks:
            x, logdet = block(x, cond)
            logdets = logdets + logdet
            outputs.append(x)
        return x, outputs, logdets
