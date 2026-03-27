# AdverAttackProj

Adversarial attacks and defenses for learned image compression models. This project evaluates the robustness of neural image compression architectures (from the CompressAI library) against gradient-based adversarial perturbations, and explores pre-compression defense strategies to mitigate the degradation they cause.

---

## Contributors

**Project Head:** Razan Dibo
**Students:** Viktor Vasilev, Danila Vodopyanov, Anastasija Cvijovic, Stepan Epifantsev
---

## Project Overview

Learned image compression replaces hand-crafted codecs (JPEG, WebP) with end-to-end neural networks that jointly optimize an encoder, a quantizer, an entropy model, and a decoder. While these models achieve state-of-the-art rate-distortion performance, they are differentiable — which makes them vulnerable to adversarial perturbations.

This project investigates three questions:

1. **Baseline performance** — How do different learned compression architectures compare on standard rate-distortion metrics across quality levels? (see `baseline_experiments/`)
2. **Attack effectiveness** — How much can small, imperceptible input perturbations degrade reconstruction quality or inflate bitrate? (see `attack_experiments/`)
3. **Defense feasibility** — Can simple input-space preprocessing (blurring, JPEG pre-compression, bit-depth reduction) reduce attack effectiveness without unacceptable quality loss? (see `defense_experiments/`)

---

## Repository Structure

```
AdverAttackProj/
├── README.md
├── requirements.txt
├── .gitignore
│
├── attack_experiments/            # Systematic attack evaluation pipeline
│   ├── attacks.py                 # FGSM, PGD, and APGD attack implementations
│   ├── metrics.py                 # MSE, PSNR, MS-SSIM, and BPP metric functions
│   ├── config.py                  # Central configuration (models, losses, epsilons, attacks)
│   ├── utils.py                   # Data loading, model loading, evaluation, CLI parsing
│   ├── run_attacks.py             # Main entry point — runs all attack combinations
│   └── attack_results_analysis.ipynb  # Notebook for analyzing saved attack results
│
├── baseline_experiments/          # Baseline rate-distortion evaluation
│   ├── baseline_experiments.ipynb       # Initial exploration (factorized, hyperprior, mbt2018)
│   └── baseline_experiments_final.ipynb # Final version (adds cheng2020 models, publication plots)
│
└── defense_experiments/           # Pre-compression defense evaluation
    ├── AdvAttck_defense1_attack2_toupload.ipynb  # Full defense-before-attack pipeline
    ├──defense_before_attacks     # Placeholder/config file
    ├──defense_before_attack_cheng2020-anchor.csv     #File with all calculated metrics on anchor model
    └── defense_for_graphs.ipynb #Graphs
```

---

## Compression Models

All models come from the [CompressAI](https://github.com/InterDigitalInc/CompressAI) zoo with pretrained weights. The project evaluates four architectures:

| Key in code | CompressAI function | Paper | Architecture summary |
|---|---|---|---|
| `bmshj_factorized` | `bmshj2018_factorized` | Ballé et al. 2018 | Factorized entropy model; simplest architecture. Encoder → quantize → factorized prior → decoder. |
| `bmshj_hyperprior` | `bmshj2018_hyperprior` | Ballé et al. 2018 | Adds a hyperprior network that transmits side information to better model the latent distribution. |
| `cheng_anchor` | `cheng2020_anchor` | Cheng et al. 2020 | Residual blocks with anchor-based context model; stronger compression at similar rates. |
| `cheng_attn` | `cheng2020_attn` | Cheng et al. 2020 | Adds attention modules to the Cheng architecture for adaptive spatial processing. |

Additional models used only in the baseline notebooks: `mbt2018` and `mbt2018_mean` (Minnen, Ballé, Toderici 2018).

All models accept input tensors of shape `(1, 3, H, W)` with pixel values in `[0, 1]` and return a dictionary containing `"x_hat"` (the reconstructed image) and `"likelihoods"` (used to compute bitrate).

**Quality parameter:** Controls the rate-distortion trade-off. Higher quality = higher bitrate + higher reconstruction fidelity. The `bmshj` and `mbt` models support quality levels 1–8; the `cheng` models support 1–6.

---

## Dataset

**Kodak24** — A standard benchmark of 24 uncompressed PNG images at 768×512 (or 512×768) resolution. Images are named `kodim01.png` through `kodim24.png`.

Loading is handled by `utils.load_kodak(path)`, which reads all 24 images, converts to RGB, and transforms to `torch.Tensor` in `[0, 1]` range.

---

## Evaluation Metrics

Defined in `attack_experiments/metrics.py`:

| Metric | Function | What it measures | Direction |
|---|---|---|---|
| **MSE** | `compute_mse(x, out)` | Mean squared error between original `x` and reconstruction `out["x_hat"]` | Lower is better |
| **PSNR** | `compute_psnr(x, out)` | Peak signal-to-noise ratio, derived as `-10 * log10(MSE)` | Higher is better |
| **MS-SSIM** | `compute_msssim(x, out)` | Multi-scale structural similarity (uses `pytorch_msssim.ms_ssim`) | Higher is better (range 0–1) |
| **BPP** | `compute_bpp(x, out)` | Bits per pixel, estimated from the entropy model's likelihoods: `sum(-log2(likelihoods)) / num_pixels` | Lower is better |

The baseline notebooks additionally compute **real BPP** via actual entropy coding (`model.compress()` → count bitstream bytes), and **SSIM** (single-scale, via scikit-image).

---

## Attack Methods

All attacks are implemented in `attack_experiments/attacks.py`. They are white-box, gradient-based, and operate in the input pixel space. Each attack perturbs the input image `x` to maximize a chosen loss function while staying within an L∞ epsilon-ball.

### FGSM (Fast Gradient Sign Method)

```
x_adv = x + eps * sign(∇_x L(x, model(x)))
```

A single-step attack. Computes the gradient of the loss with respect to the input, takes its sign, and adds a perturbation of magnitude `eps`. Fast but relatively weak.

### PGD (Projected Gradient Descent)

```
for each step:
    x_adv = x_adv + step_size * sign(∇_x L(x, model(x_adv)))
    x_adv = clip(x_adv, x - eps, x + eps)
    x_adv = clip(x_adv, 0, 1)
```

An iterative version of FGSM. Uses `step_size = eps / 4` and runs for 20 steps by default. Stronger than FGSM because it takes multiple gradient steps.

### APGD (Auto-PGD / Momentum PGD)

```
g = 0
for each step:
    grad = ∇_x L(x, model(x_adv))
    g = beta * g + grad / mean(|grad|)
    x_adv = x_adv + alpha * sign(g)
    x_adv = clip to eps-ball and [0, 1]
```

Adds momentum (`beta = 0.75`) and gradient normalization to PGD. The step size `alpha` defaults to `2 * eps / steps`. This helps escape poor local optima and generally finds stronger adversarial examples.

**Important implementation detail:** All three attacks call `model.train()` before the forward pass (to enable gradient flow through batch-norm and similar layers) and `model.eval()` afterward.

### Epsilon Values

The attack pipeline tests five perturbation budgets: `[0, 2/255, 4/255, 8/255, 16/255]`. An epsilon of 0 serves as the clean baseline. At `8/255 ≈ 0.031`, perturbations are generally imperceptible to the human eye.

---

## Loss Functions (Attack Objectives)

Defined in `attack_experiments/config.py`. These determine *what* the adversarial perturbation tries to maximize:

| Loss key | Definition | Attack goal |
|---|---|---|
| `mse` | `MSE(x, x_hat)` | Maximize reconstruction error — make the decoded image look as different from the original as possible |
| `mssim` | `1 - MS-SSIM(x, x_hat)` | Maximize perceptual distortion — degrade structural similarity |
| `bpp` | `BPP(out)` | Maximize bitrate — force the entropy model to use more bits, inflating file size |

The baseline notebooks also implement a **PGD-style composite attack** (`run_attack` in `baseline_experiments_final.ipynb`) with configurable lambdas that can simultaneously target distortion and bitrate:

```
loss = λ_dist * MSE(x_hat, x_clean) + λ_rate * BPP - λ_input * MSE(x_adv, x_clean)
```

The `λ_input` term penalizes large input perturbations as a regularizer.

---

## Defense Strategies

Implemented in `defense_experiments/AdvAttck_defense1_attack2_toupload.ipynb`. All defenses are applied **before** compression (i.e., to the input image) as a preprocessing step. The idea is that smoothing or quantizing the input can destroy the carefully crafted adversarial perturbation.

| Defense | Parameters | How it works |
|---|---|---|
| **Gaussian blur** | kernel 3×3 (σ=1.0) or 5×5 (σ=1.5) | Convolves the image with a Gaussian kernel, smoothing out high-frequency adversarial noise |
| **Median filter** | kernel 3×3 or 5×5 | Replaces each pixel with the median of its neighborhood; effective against salt-and-pepper-style noise |
| **JPEG pre-compression** | quality 75 or 90 | Saves the image as JPEG and reloads it; the lossy compression destroys small perturbations |
| **Bit-depth reduction** | 4-bit or 6-bit | Quantizes pixel values to fewer discrete levels, rounding away small perturbations |
| **Unsharp mask (sharpening)** | τ=0.05, σ=1.0 | `sharpened = img + τ * (img - blurred)`. Enhances edges while suppressing smooth-region noise |

### Defense-Before-Attack Evaluation Pipeline

The defense experiment follows this pipeline for every (defense, attack, epsilon) combination:

1. Load a clean Kodak image
2. Apply the defense to produce `img_defended`
3. Run the adversarial attack on `img_defended` to produce `x_adv`
4. Compress `x_adv` through the model
5. Measure PSNR, SSIM, and BPP of the reconstruction vs. the **original clean image**
6. Compare against: (a) clean image compressed without defense, and (b) defended image compressed without attack

This produces a trade-off analysis: how much quality does the defense cost on clean images vs. how much attack degradation does it prevent.

---

## Experiment Pipelines

### 1. Baseline Experiments (`baseline_experiments/`)

Evaluates compression models without any attacks.

- Loads all 24 Kodak images
- Runs each model at multiple quality levels (1–8 for bmshj/mbt, 1–6 for cheng)
- Computes PSNR, SSIM, estimated BPP, and real BPP (via actual entropy coding)
- Generates rate-distortion curves (BPP vs. PSNR) and architecture comparison plots
- Exports results to CSV files

The `baseline_experiments_final.ipynb` notebook is the definitive version and includes all four target architectures plus publication-quality plots.

### 2. Attack Experiments (`attack_experiments/`)

Systematic grid search over all (loss × attack × epsilon × image) combinations.

**Pipeline (`run_attacks.py`):**

1. Parse CLI arguments for model name, data path, and device
2. Load all 24 Kodak images
3. Load the specified pretrained model at quality=6
4. For each loss function (MSE, MS-SSIM, BPP):
   - For each attack (FGSM, PGD, APGD):
     - For each epsilon (0, 2/255, 4/255, 8/255, 16/255):
       - For each image:
         - Generate adversarial example (skip if eps=0)
         - Evaluate all metrics on the adversarial input
         - Save the adversarial image as PNG
         - Record results
5. Periodically save intermediate results as pickle files
6. Save final results as `results_{MODEL_NAME}_all.pkl`

Total runs per model: 3 losses × 3 attacks × 5 epsilons × 24 images = **1,080 evaluations**.

### 3. Defense Experiments (`defense_experiments/`)

Tests 10 defense variants (including "no defense") against 3 attacks at 2 epsilon levels, on all 24 Kodak images.

- Uses `cheng2020-attn` at quality=4 as the target model
- Generates per-defense trade-off scatter plots (clean quality cost vs. attack mitigation)
- Saves all intermediate images (original, defended, adversarial, reconstructed) for visual inspection
- Exports results to CSV

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Download the Kodak24 dataset (24 PNG images) and place them in a `data/` directory at the project root.

### Running the Attack Pipeline

```bash
cd attack_experiments

python run_attacks.py --model bmshj_factorized --data_path ../data --device cpu

python run_attacks.py --model cheng_attn --data_path ../data --device cuda
```

**Available model choices:** `bmshj_factorized`, `bmshj_hyperprior`, `cheng_anchor`, `cheng_attn`

Results are saved to `attack_experiments/results/{MODEL_NAME}/`:

- `results_{MODEL_NAME}_all.pkl` — final pickle with all results
- `{loss}/{attack}/eps_{eps}/kodimXX.png` — adversarial images organized by loss/attack/epsilon

### Running the Baseline Experiments

Open either notebook in Jupyter:

```bash
jupyter notebook baseline_experiments/baseline_experiments_final.ipynb
```

Adjust `KODAK_DIR` to point to your local image directory. The notebooks were originally run on Kaggle.

### Running the Defense Experiments

Open the defense notebook in Jupyter:

```bash
jupyter notebook defense_experiments/AdvAttck_defense1_attack2_toupload.ipynb
```

Adjust the `load_kodak(path='data/')` path and the device setting (`mps`, `cuda`, or `cpu`).
