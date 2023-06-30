ckpt=$1
lang=$2
lenpen=$3
fairseq-generate data/mustc/en-$lang \
  --user-dir cress \
  --config-yaml config.yaml --gen-subset tst-COMMON --task speech_to_text_modified \
  --path $ckpt \
  --max-source-positions 900000 \
  --max-tokens 2000000 --beam 8 --lenpen $lenpen --scoring sacrebleu
