#!/bin/bash
# RunPod Setup Script — nanoGPT_RH (Jan 2026)
# Полный пайплайн: установка → тренировка → extraction → calibration

set -e  # Остановиться при ошибке

echo "═══════════════════════════════════════════════════════════════"
echo "   nanoGPT_RH RunPod Setup (Jan 2026)"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════
# 1. УСТАНОВКА ЗАВИСИМОСТЕЙ
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[1/6] Установка зависимостей..."
pip install --quiet torch numpy scipy matplotlib rich wandb pysr

# ═══════════════════════════════════════════════════════════════
# 2. ПРОВЕРКА GPU
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[2/6] Проверка GPU..."
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability()
    compile_ok = '✅' if cap[0] >= 8 else '❌'
    print(f'GPU: {name}')
    print(f'VRAM: {vram:.1f} GB')
    print(f'Compute: SM {cap[0]}.{cap[1]}')
    print(f'torch.compile: {compile_ok} (needs SM >= 8.0)')
else:
    print('❌ GPU не найден!')
    exit(1)
"

# ═══════════════════════════════════════════════════════════════
# 3. ТРЕНИРОВКА E4 (POSTFIX + ID-Detox)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[3/6] Тренировка SpacingMDN+POSTFIX (E4 режим)..."

# Выбор датасета
DATA_DIR="${DATA_DIR:-data/continuous_500M}"
OUT_DIR="${OUT_DIR:-out/E5_runpod}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-512}"

echo "  DATA_DIR: $DATA_DIR"
echo "  OUT_DIR: $OUT_DIR"
echo "  MAX_STEPS: $MAX_STEPS"
echo "  BATCH_SIZE: $BATCH_SIZE"

python src/train_mdn_postfix.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --seed 7 \
    --data-mode auto \
    --use-compile \
    --use-wandb \
    --wandb-project nanoGPT-RH \
    --slot-id-mode permute_per_batch \
    --use-aux-loss \
    --early-stop --patience 800 \
    --max-steps "$MAX_STEPS" \
    --eval-every 500 \
    --batch-size "$BATCH_SIZE" \
    --use-amp

echo "✅ Тренировка завершена!"

# ═══════════════════════════════════════════════════════════════
# 4. OPERATOR EXTRACTION
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[4/6] Извлечение оператора из attention..."

python scripts/extract_operator.py \
    --checkpoint "$OUT_DIR/best.pt" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR/operator_extraction" \
    --n-samples 10000 \
    --device cuda

echo "✅ Operator extraction завершён!"

# ═══════════════════════════════════════════════════════════════
# 5. CONFORMAL CALIBRATION
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[5/6] Conformal калибровка интервалов..."

python scripts/conformal_calibrate.py \
    --ckpt "$OUT_DIR/best.pt" \
    --data-dir "$DATA_DIR" \
    --alpha 0.1 \
    --output "$OUT_DIR/calibrator.json"

echo "✅ Калибровка завершена!"

# ═══════════════════════════════════════════════════════════════
# 6. ИТОГИ
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "   ГОТОВО!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Результаты в $OUT_DIR:"
echo "  - best.pt                 — лучшая модель"
echo "  - operator_extraction/    — kernel визуализация"
echo "  - calibrator.json         — conformal поправка"
echo ""
echo "Скачать результаты:"
echo "  runpodctl send $OUT_DIR"
echo ""
echo "W&B dashboard:"
echo "  wandb sync wandb/offline-run-*"
echo ""
