# CRESS: Understanding and Bridging the Modality Gap for Speech Translation

**Qingkai Fang, Yang Feng\* | Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS)**

This is a PyTorch implementation of the **ACL 2023 main conference paper** [Understanding and Bridging the Modality Gap for Speech Translation](https://arxiv.org/abs/2305.08706).

🙌 We provide our **code**, **ST/MT model weights**, and **processed ST/MT data** in this repository.

👀 Also see our other works dedicated to **bridging the modality gap** for speech translation (ST):

- [STEMM (ACL 2022)](https://aclanthology.org/2022.acl-long.486/)
- [CMOT (ACL 2023)](https://arxiv.org/abs/2305.14635)



## Table of Contents

[TOC]

## Release

We have released the following assets for **all 8 translation directions of MuST-C**:

- Processed ST data in `.tsv` format
- Processed external MT data in fairseq binary format
- SentencePiece vocabulary
- Pretrained MT models in both `base` and `expand` settings
- Pretrained CRESS models in both `base` and `expand` settings

|                       | Link                                            | Password |
| --------------------- | ----------------------------------------------- | -------- |
| **Processed ST Data** | https://pan.baidu.com/s/1J7BgcbSNwma4SdJfHENRdg | 94wu     |
| **Processed MT Data** | https://pan.baidu.com/s/1gDMOU35_pug73y0kd-F3vw | 6tbk     |
| **Vocabulary**        | https://pan.baidu.com/s/13ucCEVzAdxRu99bdZ2oIdw | nph3     |
| **MT Model (base)**   | https://pan.baidu.com/s/1xm6myQfY-wYS4D0_rMBT_g | tm6k     |
| **MT Model (expand)** | https://pan.baidu.com/s/1byufAhoYQmgA8DCf9WUZQg | 61g4     |
| **CRESS Model (base)**   | https://pan.baidu.com/s/1_KCS_-a_Ss4Bm40dTQc6Vw | ra8j     |
| **CRESS Model (expand)** | https://pan.baidu.com/s/1zGJKmJf8TEnwBLzpOmfGYQ | ctyf     |



## Environment Configuration

1. Clone this repository:

```
git clone git@github.com:ictnlp/CRESS.git
cd CRESS/
```

2. Install `fairseq`:

```
cd fairseq/
pip install --editable ./
python setup.py build develop
```

3. We organize our implementation as fairseq plug-ins in the  `cress` directory:

```
.
├── criterions
│   ├── __init__.py
│   ├── speech_and_text_translation_criterion.py
│   ├── speech_and_text_translation_with_oracle_reg_adaptive_criterion.py
│   └── speech_and_text_translation_with_oracle_reg_criterion.py
├── datasets
│   ├── audio_utils.py
│   ├── __init__.py
│   ├── speech_and_text_translation_dataset.py
│   └── speech_to_text_dataset.py
├── __init__.py
├── models
│   ├── hubert_transformer.py
│   └── __init__.py
├── tasks
│   ├── __init__.py
│   ├── speech_and_text_translation.py
│   └── speech_to_text_modified.py
├── test_scripts
│   ├── avg_epoch.sh
│   ├── test.en-x.mt.sh
│   └── test.en-x.st.sh
└── train_scripts
    ├── train.en-x.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt.cress_adaptive.sh
    ├── train.en-x.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt.cress.sh
    ├── train.en-x.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt.sh
    ├── train.en-x.postln.wmt_pretrain.mustc_mt_pretrain.sh
    └── train.en-x.postln.wmt_pretrain.sh
```

You can import our implementation with `--user-dir cress` in fairseq.



## Data Preparation

1. Make directories to store ST (MuST-C) and MT (WMT) datasets. Please specify the target language via `$TGT_LANG`:

```
TGT_LANG=de
MUSTC_ROOT=data/mustc
WMT_ROOT=data/wmt
mkdir -p $MUSTC_ROOT $WMT_ROOT
```

2. Download the [MuST-C v1.0](https://ict.fbk.eu/must-c/) archive to the `$MUSTC_ROOT` directory and uncompress it:

```
cd $MUSTC_ROOT
tar -xzvf MUSTC_v1.0_en-${TGT_LANG}.tar.gz
```

3. We provide the processed ST data and the SentencePiece vocabulary files. You can download them via the Baidu Netdisk:

|                       | Link                                            | Password |
| --------------------- | ----------------------------------------------- | -------- |
| **Processed ST Data** | https://pan.baidu.com/s/1J7BgcbSNwma4SdJfHENRdg | 94wu     |
| **Vocabulary**        | https://pan.baidu.com/s/13ucCEVzAdxRu99bdZ2oIdw | nph3     |

Put the downloaded files in the `$MUSTC_ROOT/en-${TGT_LANG}/` directory. It should look like the this:

```
.
├── binary
├── config.yaml
├── data
├── dev.tsv
├── docs
├── spm_unigram10000.model
├── spm_unigram10000.txt
├── spm_unigram10000.vocab
├── train.tsv
└── tst-COMMON.tsv
```

4. For MT pretraining, we need additional MT datasets. We provide the processed MT data in the fairseq binary format. You can download them via the Baidu Netdisk:

|                       | Link                                            | Password |
| --------------------- | ----------------------------------------------- | -------- |
| **Processed MT Data** | https://pan.baidu.com/s/1gDMOU35_pug73y0kd-F3vw | 6tbk     |

Put the downloaded files in the `$WMT_ROOT/en-${TGT_LANG}` directory.



## Model Training

The modal training contains two steps: MT pretraining and ST finetuning.

- In the `base` setting, we pretrain the model with `<transcription, translation>` pairs from the MuST-C dataset. 
- In the `expand` setting, we first pretrain the model with external MT datasets, and then pretrain the model with `<transcription, translation>` pairs from MuST-C.

All the training scripts below are configured to run using **4 GPUs**. You can adjust `--update-freq` depending on the number of your available GPUS.

Before training, please download the [HuBERT-Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) model and place it in the `checkpoints/hubert_base_ls960.pt` path.

### MT Pretraining

1. (Optional) Pretrain the model with the external MT dataset. Please run the script:

```
sh cress/train_scripts/train.en-x.postln.wmt_pretrain.sh $TGT_LANG
```

You should adjust the maximum training steps (`--max-update`) based on the size of the training data.

After training, please average the last 5 checkpoints:

```
python scripts/average_checkpoints.py \
    --inputs checkpoints/en-$tgt.postln.wmt_pretrain \
    --num-epoch-checkpoints 5 \
    --output checkpoints/$ckpt/avg_last_5_epoch.pt
```

2. Pretrain the model with `<transcription, translation>` pairs from the MuST-C dataset. Please run the script:

```
sh cress/train_scripts/train.en-x.postln.wmt_pretrain.mustc_mt_pretrain.sh $TGT_LANG
```

After training, please average the last 10 checkpoints. You can use the script `cress/test_scripts/avg_epoch.sh`. The averaged checkpoint will be used to intialize the ST model.

**To ensure consistent performance, we have released our checkpoints of pretrained MT models in both `base` and `expand` settings. You can download them via the Baidu Netdisk.**

|                 | Link                                            | Password |
| --------------- | ----------------------------------------------- | -------- |
| **MT (base)**   | https://pan.baidu.com/s/1xm6myQfY-wYS4D0_rMBT_g | tm6k     |
| **MT (expand)** | https://pan.baidu.com/s/1byufAhoYQmgA8DCf9WUZQg | 61g4     |

### Multitask Learning

1. For multitask learning (the `MTL` baseline in the paper), please run the script:

```
sh cress/train_scripts/train.en-x.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt.sh $TGT_LANG
```

### Cross-modal Regularization with Scheduled Sampling (CRESS)

1. For the `CRESS` training, please first run the script below. Note that token-level adaptive training is not used for the first 20 epochs of training.

```
sh cress/train_scripts/train.en-x.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt.cress.sh $TGT_LANG
```

2. For the subsqeuent training epochs, token-level adaptive training will be used. Please run the script:

```
sh cress/train_scripts/train.en-x.postln.wmt_pretrain.mustc_mt_pretrain.mustc_st+mt.cress_adaptive.sh $TGT_LANG
```

We also released checkpoints of CRESS. You can download and evaluate them.

|                 | Link                                            | Password |
| --------------- | ----------------------------------------------- | -------- |
| **CRESS (base)**   | https://pan.baidu.com/s/1_KCS_-a_Ss4Bm40dTQc6Vw | ra8j     |
| **CRESS (expand)** | https://pan.baidu.com/s/1zGJKmJf8TEnwBLzpOmfGYQ | ctyf     |



## Evaluation

For evaluation, please first average the last 10 checkpoints using the `cress/test_scripts/avg_epoch.sh` script. Next, please use the scripts below to evaluate the ST/MT performance of the averaged checkpoint.

The values of `--lenpen` vary across different target languages as follows:

| TGT_LANG   | De   | Fr   | Es   | Ro   | Ru   | It   | Pt   | Nl   |
| ---------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| `--lenpen` | 1.2  | 1.8  | 0.6  | 1.4  | 0.8  | 1.0  | 1.4  | 1.0  |

### ST Evaluation

To evaluation the ST performance of the model, please use the `cress/test_scripts/test.en-x.st.sh` script:

```
sh cress/test_scripts/test.en-x.st.sh $CKPT $TGT_LANG $LENPEN
```

### MT Evaluation

To evaluation the MT performance of the model, please use the `cress/test_scripts/test.en-x.mt.sh` script.

```
sh cress/test_scripts/test.en-x.mt.sh $CKPT $TGT_LANG $LENPEN
```



## Citation

In this repository is useful for you, please cite as:

```
@inproceedings{fang-and-feng-2023-understanding,
	title = {Understanding and Bridging the Modality Gap for Speech Translation},
	author = {Fang, Qingkai and Feng, Yang},
	booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
	year = {2023},
}
```
