import os
import pickle
from PIL import Image
import logging
import argparse 

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

def get_model(device, model_name, quality=6):
    model_dict = {
        "bmshj_factorized": bmshj2018_factorized,
        "bmshj_hyperprior": bmshj2018_hyperprior,
        "cheng_anchor": cheng2020_anchor,
        "cheng_attn": cheng2020_attn,
    }

    model = model_dict[model_name](quality=quality, pretrained=True).to(device)
    model.eval()

    return model

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on CompressAI models")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["bmshj_factorized", "bmshj_hyperprior", "cheng_anchor", "cheng_attn"],
        help="Model to run attacks on"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory with images"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run attacks on"
    )

    return parser.parse_args()

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def save_pickle(file_path, results_attack):
    logging.info(f"Saving results to {file_path}")
    with open(file_path, "wb") as f:
        pickle.dump(results_attack, f)
