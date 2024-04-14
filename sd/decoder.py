import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from sd.attention import SelfAttention


class VAE_Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    def forward(self, x):
        # batchsize, channels, height, width
        residue = x
        n, c, h, w = x.shape
         # batchsize, channels, height, width -> batchsize, channels, height * width
        x = x.view(n, c, h*w)
         # batchsize, channels, height * width -> batchsize, height * width, channels FLIP LAST AND SECOND TO LAST POSITION
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2) # FLIP BACK LAST AND SECOND TO LAST POSITION
        x = x.view((n,c, h, w))
        x += residue     # SKIP LAYER
        return x



class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super.__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        residue = x
        x = self.groupnorm_1(x)

        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + self.residual_layer(residue)