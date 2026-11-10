#!/bin/bash
# Validacao cruzada por grupos de episodios: treina um adaptador LoRA por dobra e mede no grupo de teste.
# Antes: python cv_splits.py (gera splits_cv0..4.json). Precisa de GPU.
cd "$(dirname "$0")"
for i in 0 1 2 3 4; do
  SPLITS=splits_cv$i.json RESULT=resultado_cv$i.json \
    python finetune.py --max-minutes 30 --out adapter_cv$i > log_cv$i.txt 2>&1
done
