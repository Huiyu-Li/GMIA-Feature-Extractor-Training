
import torch
import torch.nn as nn

def InstanceNorm2D_wrap(input_channels, momentum=0.1, affine=True,
                        track_running_stats=False, **kwargs):
    """ 
    Wrapper around default Torch instancenorm
    """
    instance_norm_layer = nn.InstanceNorm2d(input_channels, 
        momentum=momentum, affine=affine,
        track_running_stats=track_running_stats)
    return instance_norm_layer

def InstanceNorm1D_wrap(input_channels, momentum=0.1, affine=True,
                        track_running_stats=False, **kwargs):
    """ 
    Wrapper around default Torch instancenorm
    """
    instance_norm_layer = nn.InstanceNorm1d(input_channels, 
        momentum=momentum, affine=affine,
        track_running_stats=track_running_stats)
    return instance_norm_layer

def ChannelNorm2D_wrap(input_channels, momentum=0.1, affine=True,
                       track_running_stats=False, **kwargs):
    """
    Wrapper around Channel Norm module
    """
    channel_norm_layer = ChannelNorm2D(input_channels, 
        momentum=momentum, affine=affine,
        track_running_stats=track_running_stats)

    return channel_norm_layer

def ChannelNorm1D_wrap(input_channels, momentum=0.1, affine=True,
                       track_running_stats=False, **kwargs):
    """
    Wrapper around Channel Norm module
    """
    channel_norm_layer = ChannelNorm1D(input_channels, 
        momentum=momentum, affine=affine,
        track_running_stats=track_running_stats)

    return channel_norm_layer

class ChannelNorm2D(nn.Module):
    """ 
    Similar to default Torch instanceNorm2D but calculates
    moments over channel dimension instead of spatial dims.
    Expects input_dim in format (B,C,H,W)
    """

    def __init__(self, input_channels, momentum=0.1, eps=1e-3,
                 affine=True, **kwargs):
        super(ChannelNorm2D, self).__init__()

        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x):
        """
        Calculate moments over channel dim, normalize.
        x:  Image tensor, shape (B,C,H,W)
        """
        mu, var = torch.mean(x, dim=1, keepdim=True), torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mu) * torch.rsqrt(var + self.eps)

        if self.affine is True:
            # [1, 60, 1, 1], [10, 60, 256, 256], [1, 60, 1, 1]
            x_normed = self.gamma * x_normed + self.beta
        return x_normed

class ChannelNorm1D(nn.Module):
    """ 
    Similar to default Torch instanceNorm1d but calculates
    moments over channel dimension instead of spatial dims.
    Expects input_dim in format (B,C,L)
    """

    def __init__(self, input_channels, momentum=0.1, eps=1e-3,
                 affine=True, **kwargs):
        super(ChannelNorm1D, self).__init__()

        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1))

    def forward(self, x):
        """
        Calculate moments over channel dim, normalize.
        x:  Tensor, shape (B,C,L)
        """
        mu = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mu) * torch.rsqrt(var + self.eps)

        if self.affine:
            x_normed = self.gamma * x_normed + self.beta

        return x_normed