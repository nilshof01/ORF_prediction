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

def erase_columns(imgs, num_columns=2):
    # Determine the height and width of the image
    height, width = imgs.shape[2], imgs.shape[3]

    # Create a parabolic distribution for likelihood of column erasure
    x = torch.linspace(-15, 15, width, device=imgs.device)
    a = 0.0014888176096  # these coefficients are examples and would need to be calculated 
    b = 0
    c = 0.0152831145355
    parabola = a * (x - b)**2 + c
    
    # Normalize the parabola to create a probability distribution
    parabola /= parabola.sum()
    
    # Generate num_columns random indices based on the parabolic distribution
    cols = torch.multinomial(parabola, num_columns)
    
    # Create a mask where all values are initially 1
    mask = torch.ones_like(imgs)
    
    # Set the selected columns in the mask to 0
    for col in cols:
        mask[:, :, :, col] = 0.001
    
    # Erase the selected columns in the image
    erased_imgs = imgs * mask
    
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

import torch.nn as nn

class EraseColumns(nn.Module):
    def __init__(self, num_columns=2):
        super(EraseColumns, self).__init__()
        self.num_columns = num_columns

    def forward(self, x):
        # Apply the erase_columns function to the input tensor
        x = erase_columns(x, self.num_columns)
        return x
