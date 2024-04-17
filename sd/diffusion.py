import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from sd.attention import SelfAttention, CrossAttention



class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)

            elif isinstance(layer, UNET_residualBlock):
                x = layer(x, time)

            else:
                x = layer(x)

        return x



class TimeEmbedding(nn.Module):

    def __init__(self, n_emb):
        super().__init__()
        self.linear_1 = nn.Linear(n_emb, 4 * n_emb)
        self.linear_2 = nn.Linear(n_emb * 4, n_emb)
    def forward(self, x):
        x = self.linear_1(x)

        x = F.silu(x)
        x = self.linear_2(x)
        return x

class UNET(nn.Module):
    def __init__(self, n_emb):
        super().__init__()

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_residualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_residualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_residualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_residualBlock(1280, 1280)),

            SwitchSequential(UNET_residualBlock(1280, 1280)),
            ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),

        )
        self.decoders = nn.ModuleList([

            SwitchSequential(UNET_residualBlock(2560, 1280)),

            SwitchSequential(UNET_residualBlock(2560, 1280)),

            SwitchSequential(UNET_residualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_residualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_residualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_residualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_residualBlock(1280, 1280)),

            SwitchSequential(UNET_residualBlock(1280, 1280)),
            ])
            

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)


    def forward(self, latent, context, time):
        time = self.time_embedding(time)

        output = self.unet(latent, context, time)

        output = self.final(output)

        return output