import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_encoder():

    def __init__(self):
        super().__init__(
        # Batch sieze, channel, height, width -->  batch size, 128, height, width
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        

        # Batch size, 128, height, width -> Batch size, 128, height, width
        VAE_ResidualBlock(128, 128),
        # Batch size, 128, height, width -> Batch size, 128, height, width
        VAE_ResidualBlock(128, 128),


        # Batch size, 128, height, width -> Batch size, 128, heigh / 2, width / 2
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        
         # Batch size, 128, height, width -> Batch size, 256, height / 2, width  / 2
        VAE_ResidualBlock(128, 256),
         # Batch size, 256, height, width -> Batch size, 256, height / 2, width / 2
        VAE_ResidualBlock(256, 256),

        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),   

        VAE_ResidualBlock(256, 512),

        VAE_ResidualBlock(512, 512),

        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),   

        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),

        VAE_AttentionBlock(512),

        VAE_ResidualBlock(512, 512),

        nn.GroupNorm(32, 512),

        nn.SiLU(32, 512),

        nn.Conv2d(512, 8, kernel_size=3, padding=1),
        nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        #x -> batch size , channel, height, width
        # noise -> batch size, out channels, height / 8, width /8

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # add to bottom and right 
                x = F.pad(0, 1, 0, 1)
            x = module(x)

