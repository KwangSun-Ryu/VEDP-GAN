# Generation CLI

`generation` trains the selected generative model and creates synthetic data. Run the public VEDP-GAN model with `--model-name VEDP-GAN`.

## Train And Generate With VEDP-GAN

```bash
python main.py generation \
  --model-name VEDP-GAN \
  --data-name CVA \
  --data-dir ../data/ver_06 \
  --exp-dir ./exp/generation \
  --save-dir ./output \
  --config ./config/generation/vedp_gan.toml \
  --eval-model-config-dir ./config/prediction \
  --verbose-model
```

## Run Multiple Datasets

```bash
python main.py generation \
  --model-name VEDP-GAN \
  --data-name CVA HFZ SP \
  --data-dir ../data/ver_06 \
  --exp-dir ./exp/generation \
  --save-dir ./output \
  --config ./config/generation/vedp_gan.toml \
  --eval-model-config-dir ./config/prediction
```

## Include Other Generators

```bash
python main.py generation \
  --model-name CTGAN VEDP-GAN \
  --data-name CVA \
  --data-dir ../data/ver_06 \
  --exp-dir ./exp/generation \
  --save-dir ./output
```
