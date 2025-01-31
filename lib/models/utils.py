import torch
import numpy as np
import torch.nn.functional as F


def bilinear_sampler(img, coords, mode='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    # img:    (N, H_in, C )
    # coords: (N, H_out, 1)
    # output: (N, H_out, C)
    
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)


    return img

def trilinear_sampler(vol, coords, mode='bilinear'):
    """ Wrapper for grid_sample, uses voxel coordinates """
    # vol:    (N, C, D_in, H_in, W_in)
    # coords: (N, D_out, H_out, W_out, 3)
    # output: (N, C, D_out, H_out, W_out)
    
    D, H, W = vol.shape[-3:]
    D_out, H_out, W_out = coords.shape[1:4]
    
    # Normalize coordinates
    xgrid, ygrid, zgrid = coords.split([1,1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    zgrid = 2*zgrid/(D-1) - 1

    grid = torch.cat([xgrid, ygrid, zgrid], dim=-1)
    
    # Perform trilinear sampling
    vol = F.grid_sample(vol, grid, align_corners=True)
    
    return vol
 