import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from sd.attention import SelfAttention
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, casual_mask=False):
        # x.shape = batch_size, sequence length, dimension
        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        intermediate_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        k, v, q = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        weights = q @ k.transpose(-1, 2)

        if casual_mask:
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights.masked_fill(mask, -torch.inf)

        weights /= math.sqrt(self.d_head)
        weights = F.softmax(weights, dim=-1)
        output = weights @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output