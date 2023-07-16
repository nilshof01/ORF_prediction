import torch


def erase_lines(imgs, p, L):
    """
    Args:
        imgs: Tensor of shape (batch_size, num_channels, height, width).
        p: Probability of erasing a line.
        L: Length of the erased lines, in pixels.
    Returns:
        Tensor of shape (batch_size, num_channels, height, width) with lines erased.
    """
    # Generate a binary mask indicating which pixels to erase
    mask = torch.rand(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3], device=imgs.device) < p
    mask[:, :, :L, :] = 0  # This now protects the first L rows instead of columns
    
    # Convert the mask to a float tensor before subtracting it from imgs
    mask = mask.float()
    
    # Erase the lines by setting the corresponding pixels to zero
    erased_imgs = imgs * (1.0 - mask)
    
    return erased_imgs


import numpy as np

def erase_lines(imgs, L):
    # Generate a parabolic distribution
    height, width = imgs.shape[2], imgs.shape[3]
    
    # We create a parabolic mask where the likelihood of erasure for the first
    # and last pixel is 40%, and for the center pixel is 3%.
    x = torch.linspace(-15, 15, width, device=imgs.device)
    a = 0.0014888176096  # these coefficients are examples and would need to be calculated 
    b = 0
    c = 0.0152831145355 # USE https://www.geogebra.org/graphing
    parabola = a * (x - b)**2 + c
    parabola = parabola.unsqueeze(0).unsqueeze(0).expand(imgs.shape[0], 1, height, width)
    
    # Generate a binary mask indicating which pixels to erase
    mask = torch.rand(imgs.shape[0], 1, height, width, device=imgs.device) < parabola
    mask[:, :, :L, :] = 0.0001  # This now protects the first L rows instead of columns
    
    # Convert the mask to a float tensor before subtracting it from imgs
    mask = mask.float()

    # Erase the lines by setting the corresponding pixels to zero
    erased_imgs = imgs * (1.0 - mask)

    return erased_imgs




import torch.nn as nn

class EraseLines(nn.Module):
    def __init__(self, p, L):
        super(EraseLines, self).__init__()
        self.p = p
        self.L = L
        
    def forward(self, x):
        # Apply the erase_lines function to the input tensor
        x = erase_lines(x, self.p, self.L)
        return x
