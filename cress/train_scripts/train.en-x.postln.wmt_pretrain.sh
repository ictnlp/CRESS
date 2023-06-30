tgt=$1
exp=en-$tgt.postln.wmt_pretrain
fairseq-train data/mustc/en-$tgt --text-data data/wmt/en-$tgt/mustc_wmt_en_$tgt/binary --tgt-lang $tgt \
  --user-dir cress \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 1000000 --max-tokens-text 8192 --max-update 250000 \
  --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
  --arch hubert_transformer_postln --optimizer adam --adam-betas '(0.9, 0.98)' --lr 7e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0 \
  --no-progress-bar --log-format json --log-interval 100 \
  --save-interval-updates 5000 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 1 \
  --layernorm-embedding \
  --fp16 \
  --ext-mt-training \
  --hubert-model-path checkpoints/hubert_base_ls960.pt
