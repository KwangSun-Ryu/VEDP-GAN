# Ablation Study CLI

`ablation` runs generator comparison and blending ablation for the public `VEDP-GAN` package.

## Generator comparison

```bash
python main.py ablation \
  --experiment generator_comparison \
  --data-name CVA \
  --data-dir ../data/ver_06 \
  --config-dir ./config/ablation \
  --exp-dir ./exp/ablation \
  --eval-model-config-dir ./config/prediction \
  --device-ml gpu \
  --test
```

## Run Only The VEDP-GAN Baseline

```bash
python main.py ablation \
  --experiment generator_comparison \
  --variant-slug vedp_gan \
  --data-name CVA \
  --data-dir ../data/ver_06 \
  --config-dir ./config/ablation \
  --exp-dir ./exp/ablation \
  --eval-model-config-dir ./config/prediction \
  --device-ml gpu \
  --stability-num-seeds 1
```

## Blending ablation

```bash
python main.py ablation \
  --experiment blending_ablation \
  --data-name CVA \
  --data-dir ../data/ver_06 \
  --config-dir ./config/ablation \
  --exp-dir ./exp/ablation \
  --eval-model-config-dir ./config/prediction \
  --device-ml gpu \
  --test
```

`blend_alpha_05` reuses `generator_comparison/CVA/vedp_gan` results when the settings match, instead of retraining.
