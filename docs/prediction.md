# Prediction CLI

`prediction` runs ML, SDMetrics, Utility, and DCR evaluation for generated synthetic data.

## ML Evaluation Smoke Test

```bash
python main.py prediction \
  --metric-name ML \
  --model-name VEDP-GAN \
  --data-name CVA \
  --data-dir ../data/ver_06 \
  --save-dir ./output \
  --log-dir ./result/prediction \
  --eval-model-config-dir ./config/prediction \
  --eval-model-num-trials 1 \
  --device-ml gpu \
  --test
```

## Full Metric Evaluation

```bash
python main.py prediction \
  --model-name VEDP-GAN \
  --data-name CVA \
  --data-dir ../data/ver_06 \
  --save-dir ./output \
  --log-dir ./result/prediction \
  --eval-model-config-dir ./config/prediction \
  --device-ml gpu
```

## Compare Multiple Generative Models

```bash
python main.py prediction \
  --metric-name ML SDMetrics DCR \
  --model-name CTGAN VEDP-GAN \
  --data-name CVA HFZ \
  --data-dir ../data/ver_06 \
  --save-dir ./output \
  --log-dir ./result/prediction \
  --eval-model-config-dir ./config/prediction \
  --device-ml gpu
```
