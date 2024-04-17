import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_encoder():

    def __init__(self):
        super().__init__(
        # Batch size, channel, height, width -->  batch size, 128, height, width
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
        

        # Batch size, 128, height, width -> Batch size, 128, height, width
        VAE_ResidualBlock(128, 128),
        # Batch size, 128, height, width -> Batch size, 128, height, width
        VAE_ResidualBlock(128, 128),


        # Batch size, 128, height, width -> Batch size, 128, heigh / 2, width / 2
        # Stride of 2 reduces the size by factor of 2
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        
         # Batch size, 128, height, width -> Batch size, 256, height / 2, width  / 2
        VAE_ResidualBlock(128, 256),
         # Batch size, 256, height, width -> Batch size, 256, height / 2, width / 2
        VAE_ResidualBlock(256, 256),

        # Batch size, 256, height, width -> Batch size, 256, height / 4, width / 4
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

        # takes tensor and breaks it into chunks: batch_size, channels, height / 8, width / 8 ->  2 tensors of: batch_size, channels / 2, height / 8, width / 8
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = (log_variance).exp()
        std = (variance).sqrt()

        # Sample from the latent space
        x = mean + std * noise
        x *= 0.18215
        return x
    

'''
Input Dimensions:

Input to a convolutional neural network (CNN) is a 4D tensor with dimensions [Batch size, Channels, Height, Width]. In your case, the initial input tensor dimensions might be something like [N, 3, H, W], where N is the batch size, 3 is the number of input channels (e.g., RGB image), and H and W are the height and width of the image, respectively.
First Convolutional Layer (nn.Conv2d(3, 128, kernel_size=3, padding=1)):

Channels: This layer takes an input with 3 channels and outputs 128 channels.

Spatial Dimensions: The kernel size is 3x3 with padding of 1. Padding of 1 maintains the spatial dimensions of the input, so the height and width remain unchanged.

Output Dimensions: [N, 128, H, W]

Residual Blocks (VAE_ResidualBlock(128, 128)):

These blocks typically involve several layers that do not change the height or width of the input, assuming they include padding to compensate for the kernel size. The channel size also remains unchanged since both input and output channels are specified as 128.
Output Dimensions: Still [N, 128, H, W] after each residual block.

Second Convolutional Layer (nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)):
Channels: The input and output channels are the same (128).

Spatial Dimensions: The stride of 2 reduces the spatial dimensions by half, and no padding means the reduction is strictly by half the kernel influence.
Output Dimensions: [N, 128, H/2, W/2]

Residual Blocks Increasing Channels (VAE_ResidualBlock(128, 256)):

These blocks increase the channel size from 128 to 256. Assuming no change in spatial dimensions within these blocks (they should include appropriate padding if any convolution is involved).
Output Dimensions: [N, 256, H/2, W/2]

Third Convolutional Layer (nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0)):
Channels: Remains at 256.

Spatial Dimensions: Again, a stride of 2 reduces each spatial dimension by half, and no padding focuses the reduction solely on the kernel's effective area.
Output Dimensions: [N, 256, H/4, W/4]

In summary, the shape transformations are due to:

Increasing Channels: Through specific convolutional layers or within the residual blocks.

Reducing Spatial Dimensions: By using strides greater than 1 in convolutional layers and no padding (or less padding compared to the kernel size), which effectively reduces the dimensions by focusing on central features and skipping boundaries.'''

'''
Tensor Dimensions Before Splitting:

The tensor x has dimensions [N, C, H, W] after being processed through the network, where:
N is the batch size,
C is the number of channels,
H is the height,
W is the width.

Splitting the Tensor:
    mean, log_variance = torch.chunk(x, 2, dim=1)

This line splits the tensor x into two chunks along the channel dimension (dim=1). Since the argument 2 is specified, torch.chunk divides the tensor into two equal parts, assuming C is even.
Resulting Dimensions:
Both mean and log_variance will have dimensions [N, C/2, H, W]. Here, C/2 is half the number of channels of x, as the tensor is divided equally.
Purpose of mean and log_variance:

In variational autoencoders (VAEs), the network typically outputs parameters describing a probability distribution—usually a Gaussian distribution in the simplest cases. The two parameters are:
Mean (mean): Represents the mean of the Gaussian distribution.

Log Variance (log_variance): Represents the logarithm of the variance of the Gaussian distribution. Using log variance instead of variance directly is numerically more stable, especially during backpropagation, because it helps avoid underflow and overflow issues that can occur with small or large variances.
After splitting, the mean tensor represents the mean values of the latent distribution, and the log_variance tensor (after being clamped and exponentiated) gives the variance of the latent distribution.
Additional Processing on log_variance:

    log_variance = torch.clamp(x, -30, 20): 

This line seems incorrect as it uses the original tensor x instead of log_variance. It should likely be log_variance = torch.clamp(log_variance, -30, 20).
Clamping the values between -30 and 20 ensures that the exponentiation does not result in extreme values that are too small or too large, which helps in maintaining numerical stability.

Computing Standard Deviation (std):
variance = log_variance.exp(): Exponentiates the clamped log variance to get the variance.

    std = variance.sqrt(): 
Computes the standard deviation by taking the square root of the variance.

Sampling from the Latent Space:
    x = mean + std * noise: 
    This is the reparameterization trick used in VAEs. It allows the gradient to be backpropagated through the random part of the model by sampling from the distribution defined by mean and std, scaled by a noise tensor noise. The noise tensor is usually sampled from a standard normal distribution.
Final Adjustment:
    x *= 0.18215: 
This is a scaling factor specific to the modeling context or for ensuring some form of normalization.

'''