# Generation CLI

`generation`은 지정한 생성 모델을 학습하고 합성 데이터를 생성합니다. 공개용 TADGAN은 `--model-name TADGAN`으로 실행합니다.

## TADGAN 학습 및 생성

```bash
python main.py generation \
  --model-name TADGAN \
  --data-name CVA \
  --data-dir ../data/ver_06 \
  --exp-dir ./exp/generation \
  --save-dir ./output \
  --config ./config/generation/tadgan.toml \
  --verbose-model
```

## 여러 데이터셋 실행

```bash
python main.py generation \
  --model-name TADGAN \
  --data-name CVA HFZ SP \
  --data-dir ../data/ver_06 \
  --exp-dir ./exp/generation \
  --save-dir ./output \
  --config ./config/generation/tadgan.toml
```

## 다른 generator 포함 실행

```bash
python main.py generation \
  --model-name CTGAN TADGAN \
  --data-name CVA \
  --data-dir ../data/ver_06 \
  --exp-dir ./exp/generation \
  --save-dir ./output
```
