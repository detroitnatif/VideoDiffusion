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

        self.encoders = nn.Module([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_residualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(320, 640), UNET_AttentionBlock(8, 80)),



            
            

        ])

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