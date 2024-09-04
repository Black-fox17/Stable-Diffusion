import torch
from torch import nn
from torch.nn import functioal as f
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from attention import SelfAttention


class VAE_AttentionBlock(nn.module):
    def __init__(self,channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1,channels)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        residue = x
        n,c,h,w = x .shape

        x = x.view(n,c, h * w)
        x = x.transpose(-1,-2)
        x = self.attention(x)
        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))
        x += residue

        return x
        



class VAE_ResidualBlock(nn.module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32,in_channel)
        self.conv_1 = nn.Conv2d(in_channel,out_channel, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channel)
        self.conv_2 = nn.conv2d(out_channel,out_channel, kernel_size = 3 , padding = 1)

        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.conv2d(in_channel,out_channel,kernel_size = 1, padding = 0)
    
    def forward(self,x: torch.tensor) -> torch.tensor:

        residue = x
        x = self.groupnorm_1(x)
        x = f.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = f.silu(x)
        x = self.conv2(x)

        return x + self.residual_layer(residue)
    
import torch
import torch.nn as nn

class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18215
        return self.layers(x)