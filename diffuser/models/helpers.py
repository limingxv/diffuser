import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    """
    Calculate sinusoidal positional embeddings for input sequences.

    Args:
        dim (int): Dimension of the input sequence.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

    Forward Output:
        emb (torch.Tensor): Sinusoidal positional embeddings of shape (batch_size, seq_len, dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)  # Calculate the exponential term for sinusoidal encoding
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # Generate the positional encodings
        emb = x[:, None] * emb[None, :]  # Broadcast the positional encodings across the batch dimension
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # Concatenate the sine and cosine of the positional encodings
        return emb

class Downsample1d(nn.Module):
    """
    Perform down-sampling on 1D input signals using 1D convolution.

    Args:
        dim (int): Dimension of the input signal.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, input_channels, input_length).

    Forward Output:
        y (torch.Tensor): Output tensor after down-sampling.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)  # 1D convolutional downsampling

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    """
    Perform up-sampling on 1D input signals using 1D transposed convolution.

    Args:
        dim (int): Dimension of the input signal.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, input_channels, input_length).

    Forward Output:
        y (torch.Tensor): Output tensor after up-sampling.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)  # 1D transposed convolutional upsampling

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
    1D Convolutional block with Group Normalization and Mish activation.
    Conv1d --> GroupNorm --> Mish

    Args:
        inp_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        n_groups (int): Number of groups for Group Normalization.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, inp_channels, seq_len).

    Forward Output:
        y (torch.Tensor): Output tensor after the convolutional block.
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),  # 1D convolution
            Rearrange('batch channels horizon -> batch channels 1 horizon'),  # Rearrange the tensor
            nn.GroupNorm(n_groups, out_channels),  # Group normalization
            Rearrange('batch channels 1 horizon -> batch channels horizon'),  # Rearrange the tensor back to original shape
            nn.Mish(),  # Mish activation function
        )

    def forward(self, x):
        return self.block(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    """
    Residual connection module for wrapping other modules.

    Args:
        fn (nn.Module): The module for which residual connection is applied.

    Forward Input:
        x (torch.Tensor): Input tensor.

    Forward Output:
        y (torch.Tensor): Output tensor after the residual connection.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x  # Residual connection

class LayerNorm(nn.Module):
    """
    1D Layer Normalization module.

    Args:
        dim (int): Dimension for the normalization.
        eps (float): Epsilon value for numerical stability.

    Forward Input:
        x (torch.Tensor): Input tensor.

    Forward Output:
        y (torch.Tensor): Normalized output tensor.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))  # Gain parameter
        self.b = nn.Parameter(torch.zeros(1, dim, 1))  # Bias parameter

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)  # Compute variance
        mean = torch.mean(x, dim=1, keepdim=True)  # Compute mean
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b  # Layer normalization

class PreNorm(nn.Module):
    """
    Pre-Normalization module for applying layer normalization before processing with another function.

    Args:
        dim (int): Dimension for the normalization.
        fn (nn.Module): The function to be applied after normalization.

    Forward Input:
        x (torch.Tensor): Input tensor.

    Forward Output:
        y (torch.Tensor): Output tensor after pre-normalization and function application.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    """
    1D Linear Attention module.

    Args:
        dim (int): Dimension of the input tensor.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.

    Forward Input:
        x (torch.Tensor): Input tensor.

    Forward Output:
        y (torch.Tensor): Output tensor after linear attention computation.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)  # Linear transformation for query, key, value
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)  # Linear transformation for output

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)  # Split the linear projection into query, key, and value
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)  # Softmax attention weights for key
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # Compute context

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)  # Compute output using context and query
        out = einops.rearrange(out, 'b h c d -> b (h c) d')  # Rearrange the output
        return self.to_out(out)

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)  # Gather values from tensor a using indices from tensor t
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # Reshape the output to match x_shape

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2  # Calculate the alphas cumulative products
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize the cumulative products
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # Calculate betas using the cumulative products
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)  # Clip the betas
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()  # Apply conditioning values to the input tensor
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):
    """
    Base class for computing weighted loss with additional information.

    Args:
        weights (torch.Tensor): Weights for the loss computation.
        action_dim (int): Dimension of the action space.

    Forward Input:
        pred (torch.Tensor): Predicted tensor of shape (batch_size, horizon, transition_dim).
        targ (torch.Tensor): Target tensor of shape (batch_size, horizon, transition_dim).

    Forward Output:
        weighted_loss (torch.Tensor): Weighted loss value.
        info (dict): Additional information including a0_loss.
    """

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)  # Register weights as buffer
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [batch_size x horizon x transition_dim]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()  # Compute the weighted loss
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()  # Compute the loss for the first action
        return weighted_loss, {'a0_loss': a0_loss}

class ValueLoss(nn.Module):
    """
    Base class for computing value loss with additional statistical information.

    Forward Input:
        pred (torch.Tensor): Predicted tensor.
        targ (torch.Tensor): Target tensor.

    Forward Output:
        loss (torch.Tensor): Value loss value.
        info (dict): Additional statistical information.
    """
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()  # Compute the loss

        if len(pred) > 1:
            # Calculate the correlation coefficient
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0, 1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')  # Compute the L2 loss

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')  # Compute the L2 loss

Losses = {
    'l1': WeightedL1,  # Weighted L1 loss
    'l2': WeightedL2,  # Weighted L2 loss
    'value_l1': ValueL1,  # Value L1 loss
    'value_l2': ValueL2,  # Value L2 loss
}
