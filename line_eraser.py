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
    mask[:, :, :, :L] = 0
    
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
