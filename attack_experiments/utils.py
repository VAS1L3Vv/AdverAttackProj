from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    cheng2020_attn
)

def load_kodak(path="data"):
    images = []
    names = []
    
    for i in range(1, 25):
        file = f"{path}/kodim{i:02d}.png"
        img = Image.open(file).convert("RGB")
        images.append(transforms.ToTensor()(img))
        names.append(f"kodim{i:02d}")
        
    return images, names

def get_models(device, quality=6):
    models = {
        "bmshj_factorized": bmshj2018_factorized(quality=quality, pretrained=True).to(device),
        "bmshj_hyperprior": bmshj2018_hyperprior(quality=quality, pretrained=True).to(device),
        "cheng_anchor": cheng2020_anchor(quality=quality, pretrained=True).to(device),
        "cheng_attn": cheng2020_attn(quality=quality, pretrained=True).to(device),
    }

    for name in models:
        models[name].eval()

    return models

def evaluate_model(model, x, metrics_dict):
    with torch.no_grad():
        out = model(x)

    metrics = {}
    for met, met_f in metrics_dict.items():
        metrics[met] = met_f(x, out).item()

    return out, metrics

def save_img(img, filename):
    img = img.detach().cpu()
    save_image(img, filename)