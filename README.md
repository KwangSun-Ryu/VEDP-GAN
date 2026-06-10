# VEDP-GAN

This is the official repository for the VEDP-GAN paper.

This repository contains the tabular data generation model, downstream prediction
evaluation, ablation study pipeline, and the healthcare datasets used for the
reported experiments.

## Repository Structure

```text
VEDP-GAN/
├── main.py
├── data/
│   ├── datasets_info.json
│   ├── cols_info/             # metadata for the included healthcare datasets
│   ├── original_data/         # original healthcare datasets
│   └── synthetic_data/        # VEDP-GAN synthetic datasets
├── generation/
│   ├── VEDP_GAN/              # canonical VEDP-GAN architecture
│   ├── TabDDPM/
│   ├── STaSy/
│   ├── CoDi/
│   ├── AutoDiff/
│   └── TTGAN/
├── prediction/                # downstream ML and statistical evaluation
├── ablation_study/            # ablation runners built on generation/VEDP_GAN
├── config/
    ├── generation/
    ├── prediction/
    └── ablation/
```

## Model Overview

VEDP-GAN is a tabular data generation model that combines a variational
encoder-decoder, latent-space diffusion, and adversarial generation.

The model first maps mixed tabular features into a continuous latent space with
an encoder. A forward diffusion process perturbs this latent representation, and
a conditional generator learns to synthesize latent samples from random noise,
diffusion timestep, and target label information. A discriminator operates in
the same latent space and provides both adversarial feedback and class
prediction feedback. The decoder maps latent samples back to mixed tabular
features with separate continuous and discrete output heads.

The core modules are:

- `Encoder`: maps continuous and one-hot categorical/binary features to latent
  variables `z0`, `mu`, and `logvar`.
- `Diffusion`: samples noisy latent variables `zt` from `z0`.
- `Generator`: produces synthetic latent variables conditioned on noise,
  timestep, and class label.
- `Discriminator`: distinguishes real and generated latent variables and
  predicts the target class.
- `DecoderMixed`: reconstructs continuous features and discrete feature groups.
- `VEDP-GAN`: combines the modules above into the final generator architecture.

## Losses

VEDP-GAN uses reconstruction, KL regularization, adversarial, and auxiliary
classification losses.

The reconstruction loss is computed from the decoder output:

```text
L_rec = L_cont + L_bin + L_cat
```

- `L_cont`: MSE loss for continuous columns.
- `L_bin`: binary cross-entropy with logits for binary columns.
- `L_cat`: cross-entropy loss for grouped one-hot categorical columns.

The variational regularization term is:

```text
L_KL = 0.5 * mean(exp(logvar) + mu^2 - 1 - logvar)
```

The discriminator loss is:

```text
L_D = BCE(D_adv(z_real), 1) + BCE(D_adv(z_fake), 0)
      + w_cls * CE(D_cls(z_real), y)
```

The generator adversarial loss is:

```text
L_G = BCE(D_adv(z_fake), 1) + w_cls * CE(D_cls(z_fake), y)
```

For the public VEDP-GAN configuration, the optimized generator-side objective is:

```text
L_total = w_rec * L_rec + w_kl * L_KL + L_G
```

where `z_real` is the blended latent reference:

```text
z_real = alpha * z0 + (1 - alpha) * zt
```

The default setting uses `alpha = 0.5`.

## Stage-wise Training

The current public VEDP-GAN setup uses two effective training objectives.
With the default config, Stage 1 ends at epoch `150`. Stage 2 starts after that
and remains the effective objective for the rest of training.

### Stage 1: Encoder-Decoder Warm-up

Stage 1 trains only the encoder and decoder. The generator and discriminator are
not updated in this stage.

```text
L_stage1 = w_rec * L_rec + w_kl * L_KL
```

This stage stabilizes the latent representation and teaches the decoder how to
reconstruct mixed tabular features before adversarial training starts.

### Stage 2: Latent Adversarial Generation

Stage 2 adds the latent generator and discriminator. The encoder-decoder
continues to receive reconstruction and KL gradients, while the generator learns
to produce latent samples that the discriminator classifies as real and
label-consistent.

```text
L_stage2 = w_rec * L_rec + w_kl * L_KL + L_G
```

The discriminator is trained with `L_D`, optionally with R1 regularization when
enabled in the config.

### Stage 3 Compatibility

The training code still contains a stage-3 branch for compatibility with older
regularization experiments. In the public VEDP-GAN configuration, stage-3-only
terms are disabled:

```text
latent_align_weight = 0.0
use_mode_seeking_regularization = false
mode_seeking_weight = 0.0
stage3_mode_seeking_scale = 0.0
```

Therefore, after the warm-up stage, the effective objective remains the Stage 2
objective. In other words, this repository treats VEDP-GAN as a two-objective
training procedure: reconstruction warm-up followed by latent adversarial
generation. The default config keeps `stage2_end_epoch = 500` only for
checkpoint-selection compatibility; it does not introduce an additional loss
term after epoch `500`.

## Environment

The codebase is designed for PyTorch 2.x and GPU-enabled tabular ML evaluation.
Create a conda environment from the provided environment file:

```bash
conda env create -f VEDP-GAN.yml
conda activate VEDP-GAN
```

To use a custom environment name, create the environment with `-n <env_name>`:

```bash
conda env create -n <env_name> -f VEDP-GAN.yml
conda activate <env_name>
```

If GPU evaluation is used, make sure that PyTorch, RAPIDS/cuML, XGBoost,
LightGBM, and CatBoost are installed with CUDA support compatible with the local
system.

## Data Directory

The `--data-dir <data_path>` argument should point to the root directory of the
tabular benchmark data. The expected directory layout is:

```text
<data_path>/
├── datasets_info.json
├── original_data/
│   └── <dataset_name>.csv
└── cols_info/
    └── <dataset_name>_metadata.json
```

For example, if the dataset name is `CVA`, the runner expects:

```text
<data_path>/original_data/CVA.csv
<data_path>/cols_info/CVA_metadata.json
```

## Usage

All entry points are exposed through the root `main.py`.

```bash
python main.py generation --help
python main.py prediction --help
python main.py ablation --help
```

## Example Commands

### Training and Sampling

Train VEDP-GAN on GPU, then generate 10x synthetic samples from the selected
best checkpoint.

```bash
python main.py generation \
  --model-name VEDP-GAN \
  --data-name <dataset_name> \
  --data-dir <data_path> \
  --exp-dir ./exp/generation \
  --save-dir ./output \
  --config ./config/generation/vedp_gan.toml \
  --eval-model-config-dir ./config/prediction \
  --verbose-model

python -m generation.inference \
  --model-name VEDP-GAN \
  --data-name <dataset_name> \
  --data-dir <data_path> \
  --exp-dir ./exp/generation \
  --save-dir ./output \
  --config ./config/generation/vedp_gan.toml \
  --eval-model-config-dir ./config/prediction \
  --multiplier 10 \
  --verbose-model
```

To train and sample all generation models for one dataset, list all model names
or omit `--model-name`.

```bash
python main.py generation \
  --model-name TabDDPM CTGAN STaSy CoDi AutoDiff TTGAN VEDP-GAN \
  --data-name <dataset_name> \
  --data-dir <data_path> \
  --exp-dir ./exp/generation \
  --save-dir ./output \
  --eval-model-config-dir ./config/prediction \
  --verbose-model
```

To run every generation model on every dataset in `datasets_info.json`, omit
both `--model-name` and `--data-name`.

### Evaluation

Run GPU-based ML evaluation with 10 trials.

```bash
python main.py prediction \
  --metric-name ML \
  --model-name VEDP-GAN \
  --data-name <dataset_name> \
  --data-dir <data_path> \
  --save-dir ./output \
  --log-dir ./result/prediction \
  --eval-model-config-dir ./config/prediction \
  --eval-model-num-trials 10 \
  --device-ml gpu
```

### Ablation Study

Run the full ablation workflow on GPU.

```bash
python main.py ablation \
  --experiment generator_comparison \
  --data-name <dataset_name> \
  --data-dir <data_path> \
  --config-dir ./config/ablation \
  --exp-dir ./exp/ablation \
  --eval-model-config-dir ./config/prediction \
  --eval-model-num-trials 10 \
  --device-ml gpu \
  --device-dcr gpu \
  --device-train cuda \
  --stability-num-seeds 1

python main.py ablation \
  --experiment blending_ablation \
  --data-name <dataset_name> \
  --data-dir <data_path> \
  --config-dir ./config/ablation \
  --exp-dir ./exp/ablation \
  --eval-model-config-dir ./config/prediction \
  --eval-model-num-trials 10 \
  --device-ml gpu \
  --device-dcr gpu \
  --device-train cuda \
  --stability-num-seeds 1
```

More examples are available in:

- `docs/generation.md`
- `docs/prediction.md`
- `docs/ablation_study.md`

## VEDP-GAN Configuration

The default public VEDP-GAN configuration is stored in
`config/generation/vedp_gan.toml`. Ablation configs under `config/ablation` reuse
the same implementation and vary only the components required by each ablation.

The default VEDP-GAN setting uses:

- `latent_align_weight = 0.0`
- `use_mode_seeking_regularization = false`
- `mode_seeking_weight = 0.0`
- `stage3_mode_seeking_scale = 0.0`
- `alpha = 0.5`
