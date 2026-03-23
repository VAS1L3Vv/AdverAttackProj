import os
from tqdm import tqdm
import pickle
import logging

import torch

from utils import *
from config import *

def main():
    setup_logger()

    args = parse_args()
    MODEL_NAME = args.model
    DEVICE = args.device
    DATA_PATH = args.data_path

    logging.info(f"Selected model: {MODEL_NAME}")
    logging.info(f"Running on device: {DEVICE}\n")
    base_dir = os.path.join("results", MODEL_NAME)
    os.makedirs(base_dir, exist_ok=True)

    logging.info("Loading Kodak24 dataset...")
    images, image_names = load_kodak(DATA_PATH)
    images = [img.unsqueeze(0).to(DEVICE) for img in images]
    logging.info(f"Loaded {len(images)} images\n")

    logging.info("Loading model...")
    model = get_model(DEVICE, MODEL_NAME)
    logging.info("Model loaded successfully\n")

    results_attack = []
    total_runs = (len(LOSSES) * len(ATTACKS) * len(EPSILONS) * len(images))
    logging.info(f"Total runs to execute: {total_runs}\n")

    for i_loss, (loss_name, loss_fn) in enumerate(LOSSES.items()):
        for i_at, (attack_name, attack_fn) in enumerate(ATTACKS.items()):
            for i_eps, eps in enumerate(EPSILONS):
                print("")
                logging.info(f"Starting loss: {loss_name}, {i_loss}/{len(LOSSES)}")
                logging.info(f"Attack: {attack_name}, {i_at}/{len(ATTACKS)}")
                logging.info(f"Epsilon: {eps}, {i_eps}/{len(EPSILONS)}")

                for img, name in tqdm(zip(images, image_names), total=len(images), desc="Images", leave=False):
                    if eps == 0:
                        x_adv = img.clone()
                    else:
                        x_adv, _ = attack_fn(model, img, eps=eps, loss_fn=loss_fn)

                    with torch.no_grad():
                        _, metrics = evaluate_model(model, x_adv, METRICS)

                    if eps != 0:
                        dir_path = os.path.join(base_dir, loss_name, attack_name, f"eps_{eps:.5f}")
                        os.makedirs(dir_path, exist_ok=True)
                        adv_filename = os.path.join(dir_path, f"{name}.png")
                        save_img(x_adv, adv_filename)
                    else:
                        adv_filename = os.path.join("data", f"{name}.png")

                    results_attack.append({
                        "attack": attack_name,
                        "epsilon": eps,
                        "loss" : loss_name,
                        "image": name,
                        "metrics": metrics,
                        "adv_path": adv_filename,
                    })

                    del x_adv

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results_file = os.path.join(base_dir, f"results_{MODEL_NAME}.pkl")
            save_pickle(results_file, results_attack)
    
    final_results_file = os.path.join(base_dir, f"results_{MODEL_NAME}_all.pkl")
    save_pickle(final_results_file, results_attack)
    logging.info("Code finished successfully")

if __name__ == "__main__":
    main()