import torch
from metrics import compute_psnr, compute_msssim, compute_bpp, compute_mse
from attacks import fgsm_attack, pgd_attack, apgd_attack


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUALITY = 6

METRICS = {
    "psnr": compute_psnr,
    "mssim": compute_msssim,
    "bpp": compute_bpp
}

LOSSES = {
    "mse": compute_mse,
    "psnr": lambda x, out: 1 - compute_psnr(x, out),
    "mssim": lambda x, out: 1 - compute_msssim(x, out),
    "bpp": compute_bpp
}

EPSILONS = [0, 2/255, 4/255, 8/255, 16/255, 32/255]

ATTACKS = {
    "FGSM": fgsm_attack,
    "PGD": pgd_attack,
    "APGD": apgd_attack
}