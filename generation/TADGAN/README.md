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

Generation orchestration, checkpoint selection, and sampling utilities are
restored inside `generation/TADGAN/scripts` for standalone public execution.
