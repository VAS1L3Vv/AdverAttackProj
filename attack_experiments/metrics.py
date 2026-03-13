import math
import torch
from pytorch_msssim import ms_ssim

def compute_mse(x, out):
    x_hat = out["x_hat"]
    mse = torch.mean((x - x_hat) ** 2)
    return mse

def compute_psnr(x, out):
    x_hat = out["x_hat"]
    mse = torch.mean((x - x_hat) ** 2)
    return -10 * torch.log10(mse)

def compute_msssim(x, out):
    x_hat = out["x_hat"].clamp(0,1) 
    return ms_ssim(x, x_hat, data_range=1.)


def compute_bpp(x, out):
    size = out["x_hat"].size()
    num_pixels = size[0] * size[2] * size[3]

    bpp = sum(
        torch.log(likelihoods).sum() /
        (-math.log(2) * num_pixels)
        for likelihoods in out["likelihoods"].values()
    )

    return bpp