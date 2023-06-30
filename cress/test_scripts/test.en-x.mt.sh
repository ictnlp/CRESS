ckpt=$1
lang=$2
lenpen=$3
fairseq-generate data/mustc/en-$lang --text-data data/mustc/en-$lang/binary --tgt-lang $lang \
  --user-dir cress \
  --config-yaml config.yaml --gen-subset test --task speech_and_text_translation \
  --path $ckpt \
  --ext-mt-training \
  --max-tokens 2000000 --max-tokens-text 4096 --beam 8 --lenpen $lenpen --scoring sacrebleu