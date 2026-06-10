# VEDP-GAN Model Package

`generation/VEDP_GAN` contains the canonical VEDP-GAN model implementation.

## Files

```text
generation/VEDP_GAN/
├── model.py           # VEDP-GAN architecture
└── __init__.py
```

## Architecture

`model.py` defines the complete VEDP-GAN architecture:

- `Encoder`: variational encoder for mixed tabular features
- `Diffusion`: latent forward diffusion process
- `Generator`: conditional latent generator
- `Discriminator`: auxiliary-classifier discriminator
- `DecoderMixed`: decoder with continuous and discrete output heads
- `VEDP-GAN`: end-to-end model that combines the modules above

Generation orchestration, checkpoint selection, and sampling utilities are
restored inside `generation/VEDP_GAN/scripts` for standalone public execution.
