# Ablation Study CLI

`ablation`은 공개용 `TADGAN`을 기준으로 generator comparison과 blending ablation을 실행합니다.

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

## TADGAN baseline만 실행

```bash
python main.py ablation \
  --experiment generator_comparison \
  --variant-slug tadgan \
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

`blend_alpha_05`는 `generator_comparison/CVA/tadgan` 결과와 설정이 같으면 재학습하지 않고 해당 결과를 재사용합니다.
