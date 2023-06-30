# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig, 
    label_smoothed_nll_loss,
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class SpeechAndTextTranslationOracleRegCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    reg_weight: float = field(
        default=1.0,
        metadata={"help": "weight of regularization loss"},
    )
    reg_loss_type: str = field(
        default="jsd",
        metadata={"help": "loss type of regularization (e.g. jsd, l1)"},
    )
    use_word_level_oracle: bool = field(
        default=False,
        metadata={"help": "use word level oracles"},
    )
    decay_k: float = field(
        default=15,
        metadata={"help": "decay hyper-paramter k"},
    )
    use_word_gumbel_noise: bool = field(
        default=False,
        metadata={"help": "select word with gumbel noise"},
    )
    gumbel_temperature: float = field(
        default=1.0,
        metadata={"help": "temperature of gumbel max in word oracles"},
    )

@register_criterion(
    "speech_and_text_translation_with_oracle_reg", dataclass=SpeechAndTextTranslationOracleRegCriterionConfig
)
class SpeechAndTextTranslatioOracleRegCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        reg_weight=1.0,
        reg_loss_type="jsd",
        use_word_level_oracle=False,
        decay_k=15,
        use_word_gumbel_noise=False,
        gumbel_temperature=1.0,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.reg_weight = reg_weight
        self.padding_idx = task.target_dictionary.pad()
        self.tgt_dict = task.target_dictionary
        self.bpe_tokenizer = task.bpe_tokenizer
        self.reg_loss_type = reg_loss_type
        self.use_word_level_oracle = use_word_level_oracle
        self.decay_k = decay_k
        self.use_word_gumbel_noise = use_word_gumbel_noise
        self.gumbel_temperature = gumbel_temperature
    
    def decay_prob(self, epoch):
        k = self.decay_k
        return k / (k + np.exp(epoch / k))
    
    def get_word_oracle_tokens(self, pred_logits, prev_output_tokens, epoch, epsilon=1e-6):
        bsz, _ = prev_output_tokens.size()
        if self.use_word_gumbel_noise:
            uniform = torch.Tensor(pred_logits.size()).to(pred_logits.device).float().uniform_(0, 1)
            gumbel = -torch.log(-torch.log(uniform + epsilon) + epsilon)
            pred_logits = (pred_logits + gumbel.to(pred_logits.device)) / self.gumbel_temperature
        pred_tokens = torch.max(pred_logits, dim=-1)[1]
        bos_idx = prev_output_tokens[0, 0].repeat(bsz, 1).to(pred_tokens)
        pred_tokens = torch.cat([bos_idx, pred_tokens], dim=1)[:, :-1]
        sample_gold_prob = self.decay_prob(epoch)
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        return prev_output_tokens * sample_gold_mask + pred_tokens * (1 - sample_gold_mask)

    def forward_st(self, model, sample, reduce, word_oracle=False):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        audio_encoder_out = model.encoder(
            audio_input["src_tokens"], 
            audio_input["src_lengths"], 
            audio_input["mode"]
        )
        prev_output_tokens = audio_input["prev_output_tokens"]
        with torch.no_grad():
            if word_oracle:
                audio_output = model.decoder(
                    prev_output_tokens,
                    audio_encoder_out,
                )
                prev_output_tokens = self.get_word_oracle_tokens(
                    audio_output[0].detach(),
                    prev_output_tokens,
                    model.epoch,
                )
        audio_output = model.decoder(
            prev_output_tokens,
            audio_encoder_out,
        )
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, audio_output, sample, reduce=reduce)
        return loss, lprobs, target
    
    def forward_mt(self, model, sample, reduce, word_oracle=False):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        text_encoder_out = model.encoder(
            text_input["src_tokens"], 
            text_input["src_lengths"], 
            text_input["mode"]
        )
        prev_output_tokens = text_input["prev_output_tokens"]
        with torch.no_grad():
            if word_oracle:
                text_output = model.decoder(
                    prev_output_tokens,
                    text_encoder_out,
                )
                prev_output_tokens = self.get_word_oracle_tokens(
                    text_output[0].detach(),
                    prev_output_tokens,
                    model.epoch,
                )
        text_output = model.decoder(
            prev_output_tokens,
            text_encoder_out,
        )
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, text_output, sample, reduce=reduce)
        return loss, lprobs, target
    
    def forward_ext_mt(self, model, sample, reduce):
        text_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss, ext_mt_loss, reg_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        st_size, mt_size, ext_mt_size, reg_size = 0, 0, 0, 0

        mode = sample["net_input"]["mode"]
        if mode == "st":
            if self.training:
                word_oracle = self.use_word_level_oracle
                st_loss, st_lprobs, st_target = self.forward_st(model, sample, reduce, word_oracle)
                mt_loss, mt_lprobs, mt_target = self.forward_mt(model, sample, reduce, word_oracle)
                reg_loss = self.compute_reg_loss(st_lprobs, mt_lprobs, st_target, mt_target)
                loss = st_loss + mt_loss + self.reg_weight * reg_loss
                st_size = mt_size = sample_size = reg_size = sample["ntokens"]
            else:
                st_loss, _, _ = self.forward_st(model, sample, reduce)
                loss = st_loss
                st_size = sample_size = sample["ntokens"]
        elif mode == "ext_mt":
            loss = ext_mt_loss = self.forward_ext_mt(model, sample, reduce)
            ext_mt_size = sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "reg_loss": reg_loss.data,
            "reg_sample_size": reg_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        
        return loss, sample_size, logging_output

    def compute_loss_with_lprobs(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, target
    
    def compute_jsd_loss(self, st_lprobs, mt_lprobs, st_target, mt_target, ignore_index):
        kl_loss_st = F.kl_div(mt_lprobs, st_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(st_lprobs, mt_lprobs, log_target=True, reduction="none").sum(-1)
        pad_mask = st_target.eq(ignore_index)
        kl_loss_st.masked_fill_(pad_mask, 0.0)
        pad_mask = mt_target.eq(ignore_index)
        kl_loss_mt.masked_fill_(pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        kl_loss = (kl_loss_st + kl_loss_mt) / 2.0
        return kl_loss
    
    def compute_reg_loss(self, st_lprobs, mt_lprobs, st_target, mt_target):
        if self.reg_loss_type == "jsd":
            return self.compute_jsd_loss(st_lprobs, mt_lprobs, st_target, mt_target, self.padding_idx)
        else:
            raise NotImplementedError

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        reg_loss_sum = sum(log.get("reg_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(log.get("ext_mt_sample_size", 0) for log in logging_outputs)
        reg_sample_size = sum(log.get("reg_sample_size", 0) for log in logging_outputs)        

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "ext_mt_loss", ext_mt_loss_sum / ext_mt_sample_size / math.log(2) if ext_mt_sample_size != 0 else 0, ext_mt_sample_size, round=3
        )
        metrics.log_scalar(
            "reg_loss", reg_loss_sum / reg_sample_size / math.log(2) if reg_sample_size != 0 else 0, reg_sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True