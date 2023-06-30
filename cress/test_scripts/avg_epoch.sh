ckpt=$1
python scripts/average_checkpoints.py \
    --inputs checkpoints/$ckpt \
    --num-epoch-checkpoints 10 \
    --output checkpoints/$ckpt/avg_last_10_epoch.pt