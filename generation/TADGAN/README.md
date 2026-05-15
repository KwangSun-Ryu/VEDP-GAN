# TADGAN Model Package

`generation/TADGAN` contains the canonical TADGAN model implementation.

## Files

```text
generation/TADGAN/
├── model.py           # TADGAN architecture
└── __init__.py
```

## Architecture

`model.py` defines the complete TADGAN architecture:

- `Encoder`: variational encoder for mixed tabular features
- `Diffusion`: latent forward diffusion process
- `Generator`: conditional latent generator
- `Discriminator`: auxiliary-classifier discriminator
- `DecoderMixed`: decoder with continuous and discrete output heads
- `TADGAN`: end-to-end model that combines the modules above

Experiment orchestration, ablation variants, checkpoint selection, and
evaluation utilities are intentionally kept in `ablation_study`.
