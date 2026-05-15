# TADGAN

Official research code structure for TADGAN.

This repository contains the tabular data generation model, downstream prediction
evaluation, and ablation study pipeline used for the experiments.

## Repository Structure

```text
TADGAN/
├── main.py
├── generation/
│   ├── TADGAN/              # canonical TADGAN architecture
│   ├── TabDDPM/
│   ├── STaSy/
│   ├── CoDi/
│   ├── AutoDiff/
│   └── TTGAN/
├── prediction/              # downstream ML and statistical evaluation
├── ablation_study/          # ablation runners built on generation/TADGAN
├── config/
│   ├── generation/
│   ├── prediction/
│   └── ablation/
└── docs/
```

The canonical TADGAN architecture is defined in `generation/TADGAN/model.py`.
The ablation pipeline imports this model directly, while ablation-specific
training loops, selection logic, and evaluation utilities live in
`ablation_study`.

## Model Overview

TADGAN is a tabular data generation model that combines a variational
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
- `TADGAN`: combines the modules above into the final generator architecture.

## Losses

TADGAN uses reconstruction, KL regularization, adversarial, and auxiliary
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

For the public TADGAN configuration, the optimized generator-side objective is:

```text
L_total = w_rec * L_rec + w_kl * L_KL + L_G
```

where `z_real` is the blended latent reference:

```text
z_real = alpha * z0 + (1 - alpha) * zt
```

The default setting uses `alpha = 0.5`.

## Stage-wise Training

The current public TADGAN setup uses two effective training objectives.
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
regularization experiments. In the public TADGAN configuration, stage-3-only
terms are disabled:

```text
latent_align_weight = 0.0
use_mode_seeking_regularization = false
mode_seeking_weight = 0.0
stage3_mode_seeking_scale = 0.0
```

Therefore, after the warm-up stage, the effective objective remains the Stage 2
objective. In other words, this repository treats TADGAN as a two-objective
training procedure: reconstruction warm-up followed by latent adversarial
generation. The default config keeps `stage2_end_epoch = 500` only for
checkpoint-selection compatibility; it does not introduce an additional loss
term after epoch `500`.

## Environment

The codebase is designed for PyTorch 2.x and GPU-enabled tabular ML evaluation.
Create a conda environment from the provided environment file:

```bash
conda env create -f TADGAN.yml
conda activate TADGAN
```

To use a custom environment name, create the environment with `-n <env_name>`:

```bash
conda env create -n <env_name> -f TADGAN.yml
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

Train and sample with TADGAN:

```bash
python main.py generation \
  --model-name TADGAN \
  --data-name <dataset_name> \
  --data-dir <data_path> \
  --config ./config/generation/tadgan.toml \
  --verbose-model
```

Evaluate generated data with GPU ML models:

```bash
python main.py prediction \
  --metric-name ML \
  --model-name TADGAN \
  --data-name <dataset_name> \
  --data-dir <data_path> \
  --save-dir <synthetic_data_path> \
  --eval-model-config-dir ./config/prediction \
  --eval-model-num-trials 1 \
  --device-ml gpu \
  --test
```

Run the generator ablation study:

```bash
python main.py ablation \
  --experiment generator_comparison \
  --data-name <dataset_name> \
  --data-dir <data_path> \
  --config-dir ./config/ablation \
  --eval-model-config-dir ./config/prediction \
  --device-ml gpu \
  --test
```

More examples are available in:

- `docs/generation.md`
- `docs/prediction.md`
- `docs/ablation_study.md`

## TADGAN Configuration

The default public TADGAN configuration is stored in
`config/generation/tadgan.toml`. Ablation configs under `config/ablation` reuse
the same implementation and vary only the components required by each ablation.

The default TADGAN setting uses:

- `latent_align_weight = 0.0`
- `use_mode_seeking_regularization = false`
- `mode_seeking_weight = 0.0`
- `stage3_mode_seeking_scale = 0.0`
- `alpha = 0.5`
