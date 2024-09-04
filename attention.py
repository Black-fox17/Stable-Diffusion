import torch
from torch import nn
from torch.nn import functioal as f
import math

class SelfAttention(nn.module):
    def __init__(self,n_heads,n_embed,in_proj_bias = True,out_proj_bias = True):
        super().__init__()
        self.in_proj = nn.Linear(n_embed,3 * n_embed,bias = in_proj_bias)
        self.out_proj = nn.Linear(n_embed,n_embed,bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = n_embed//n_heads

    def forward(self, x:torch.tensor , causal_mask = True):

        input_shape = x.shape
        batch_size, seq_length,d_embed = input_shape
        mid_shape = (batch_size,seq_length,self.n_heads,self,d_embed)

        q,k,v = self.in_proj(x).chunk(3, dim = -1)
        
        q = q.view(mid_shape).transpose(1,2)
        k = k.view(mid_shape).transpose(1,2)
        v = v.view(mid_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)  

        if causal_mask:
            mask = torch.ones_like(weight,stype = torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = f.softmax(weight , dims = -1)
        output = weight @ v
        output = output.reshape(input_shape)

        return output
