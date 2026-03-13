import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pickle

import torch
import torch.nn.functional as F

from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns

from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    cheng2020_attn
)

from metrics import *
from attacks import *
from utils import *
from config import *


os.makedirs("results/x_adv", exist_ok=True)
os.makedirs("results/x_hat", exist_ok=True)

images, image_names = load_kodak()
models = get_models(DEVICE)

results_attack = []

for loss_name, loss_fn in LOSSES.items():
    for attack_name, attack_fn in tqdm(ATTACKS.items()):
        for eps in EPSILONS:
            for model_name, model in models.items():
                for img, name in zip(images, image_names):
                    x = img.unsqueeze(0).to(DEVICE)

                    if eps == 0:
                        x_adv = x.clone()
                    else:
                        x_adv, _ = attack_fn(model, x, eps=eps, loss_fn=loss_fn)

                    with torch.no_grad():
                        out, metrics = evaluate_model(model, x_adv, METRICS)
                        x_hat = out["x_hat"]

                    adv_filename = f"results/x_adv/{model_name}_{attack_name}_{loss_name}_eps{eps:.5f}_{name}.png"
                    hat_filename = f"results/x_hat/{model_name}_{attack_name}_{loss_name}_eps{eps:.5f}_{name}.png"
                    

                    save_img(x_adv, adv_filename)
                    save_img(x_hat, hat_filename)

                    results_attack.append({
                        "attack": attack_name,
                        "epsilon": eps,
                        "loss" : loss_name,
                        "model": model_name,
                        "image": name,
                        "metrics": metrics,
                        "adv_path": adv_filename,
                        "hat_path": hat_filename,\
                    })

                    del x, x_adv, x_hat,out
                    torch.cuda.empty_cache()

with open("results_attack.pkl", "wb") as f:
    pickle.dump(results_attack, f)