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

            SwitchSequential(UNET_residualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_residualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_residualBlock(1920, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_residualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(960, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(960, 640), UNET_AttentionBlock(8, 80)), Upsample(640),

            SwitchSequential(UNET_residualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_residualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            ])
            

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        return x

class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_time):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)


        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature, time):
        # feature: batch_size, in_channels, height, width
        # time: 1, 1280
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_features(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNET_Attentionblock(nn.Module):
    def __init__(self, n_heads, n_emb, d_context):
        super().__init__()

        channels = n_heads * n_emb
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x , context):
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        x = x.view((n,c,h*w))
        x = x.transpose(-1, -2)

        residue_short = x
        x = self.layernorm_1(x)
        self.attention_1(x)

        x += residue_short

        residue_short = x

        x = self.layernorm_2(x)

        self.attention_2(x, context)

        x += residue_short

        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residue_short

        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long






class CrossAttention(nn.Module):
    # CROSS ATTENTION BETWEEN LATENT AND PROMPT
    def __init__(self, n_heads, d_emb, d_cross, in_bias=True, out_bias=True)
        super().__init__()
        self.q_proj = nn.Linear(d_emb, d_emb, bias=in_bias)
        self.k_proj = nn.Linear(d_cross, d_emb, bias=in_bias)
        self.v_proj = nn.Linear(d_cross, d_emb, bias=in_bias)
        self.out_proj = nn.Linear(d_emb, d_emb, bias=out_bias)

        self.n_heads = n_heads
        self.d_head = d_emb // n_heads

    def forward(self, x, y):


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