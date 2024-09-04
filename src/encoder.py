import torch
from torch import nn
from torch.nn import functioal as f
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.conv2d(3,128,kernel_size = 3,padding = 3),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            #   
            nn.conv2d(128,128,kernel_size = 3,stride = 2, padding = 0),

            VAE_ResidualBlock(128,256),
            VAE_ResidualBlock(256,256),

            nn.conv2d(256,256,kernel_size = 3,stride = 2, padding = 0),
            
            VAE_ResidualBlock(256,512),
            VAE_ResidualBlock(512,512),

            nn.conv2d(512,512,kernel_size = 3,stride = 2, padding = 0),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),

            nn.GroupNorm(32,512),
            nn.conv2d(512,8,kernel_size = 3,padding = 1),
            nn.conv2d(8,8,kernel_size = 1, padding = 0)
            )
    def forward(self, x:torch.tensor,noise: torch.tensor) -> torch.tensor:
        for module in self:
            if getattr(module, "stride",None) == (2,2):
                x = f.pad(x,(0,1,0,1))
            x = module(x)

        mean,log_variance = torch.chunk(x,2 , dim = 1)
        log_variance = torch.clamp(log_variance,-30,20)
        variance = log_variance.exp()

        std_dev = variance.sqrt()
        
        x = mean + std_dev * noise
        x *= 0.18215

        return x