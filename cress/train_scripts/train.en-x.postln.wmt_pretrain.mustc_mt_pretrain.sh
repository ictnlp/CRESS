tgt=$1
exp=en-$tgt.postln.wmt_pretrain.mustc_mt_pretrain
fairseq-train data/mustc/en-$tgt --text-data data/mustc/en-$tgt/binary/ --tgt-lang $tgt \
  --user-dir cress \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --save-dir checkpoints/${exp} --num-workers 4 --max-tokens 1000000 --max-tokens-text 8192 --max-update 100000 \
  --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
  --arch hubert_transformer_postln --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt \
  --no-progress-bar --log-format json --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --warmup-updates 8000 --clip-norm 10.0 --seed 1 --update-freq 1 \
  --layernorm-embedding \
  --patience 10 \
  --fp16 \
  --ext-mt-training \
  --hubert-model-path checkpoints/hubert_base_ls960.pt \
  --load-pretrained-mt-encoder-decoder-from checkpoints/en-$tgt.postln.wmt_pretrain/avg_last_5_epoch.pt