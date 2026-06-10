# Prediction CLI

`prediction`은 생성된 합성 데이터를 기준으로 ML, SDMetrics, Utility, DCR 평가를 실행합니다.

## ML 평가 smoke test

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

## 전체 지표 평가

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

## 여러 생성 모델 비교

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
