#!/usr/bin/env python3

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import checkpoint_utils, tasks, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler, TransformerDecoderScriptable
from fairseq.models.hubert import HubertModel
from fairseq.models.transformer import Embedding
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)

logger = logging.getLogger(__name__)

@register_model("hubert_transformer")
class HubertTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/s2t"
        model_ids = [
            "s2t_transformer_s-en-asr-librispeech",
            "s2t_transformer_m-en-asr-librispeech",
            "s2t_transformer_l-en-asr-librispeech",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config_yaml="config.yaml",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config_yaml,
            **kwargs,
        )
        return S2THubInterface(x["args"], x["task"], x["models"][0])

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.epoch = 1
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        # hubert arguments
        parser.add_argument(
            "--hubert-model-path",
            type=str,
            metavar="STR",
            help="path/to/hubert/model"
        )
        parser.add_argument(
            "--freeze-hubert",
            action="store_true",
            help="if we want to freeze the hubert features"
        )
        # subsampler arguments
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            help="# of channels in Conv1d subsampling layers",
        )
        # pretrain
        parser.add_argument(
            "--load-pretrained-mt-encoder-decoder-from",
            type=str,
            help="model to take mt encoder/decoder weight from (for initialization)",
        )

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        return HubertTransformerEncoder(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder_embed_tokens = decoder_embed_tokens
        encoder = cls.build_encoder(args, task, encoder_embed_tokens)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        # load pretrained mt models
        mt_pretrained_path = getattr(args, "load_pretrained_mt_encoder_decoder_from", None)
        if mt_pretrained_path is not None and Path(mt_pretrained_path).exists():
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(mt_pretrained_path)["model"]
            mt_encoder_state_dict = OrderedDict()
            mt_decoder_state_dict = OrderedDict()
            for key in state_dict.keys():
                if "hubert" in key or "subsampler" in key:
                    continue
                if key.startswith("encoder"):
                    subkey = key[len("encoder") + 1 :]
                    mt_encoder_state_dict[subkey] = state_dict[key]
                if key.startswith("decoder"):
                    subkey = key[len("decoder") + 1 :]
                    mt_decoder_state_dict[subkey] = state_dict[key]
            encoder.load_state_dict(mt_encoder_state_dict, strict=False)
            decoder.load_state_dict(mt_decoder_state_dict, strict=False)

        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, mode, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, mode=mode)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class HubertTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, dictionary=None, embed_tokens=None):
        super().__init__(None)

        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = dictionary.pad()

        # load hubert
        self.hubert_model_path = getattr(args, "hubert_model_path", None)
        self.freeze_hubert = getattr(args, "freeze_hubert", False)
        assert self.hubert_model_path is not None
        ckpt = checkpoint_utils.load_checkpoint_to_cpu(self.hubert_model_path)
        hubert_args = ckpt["cfg"]
        task = tasks.setup_task(hubert_args.task)
        if "task_state" in ckpt:
            task.load_state_dict(ckpt["task_state"])
        self.hubert_model = task.build_model(hubert_args.model)
        self.hubert_model.load_state_dict(ckpt["model"])
        self.hubert_model.remove_pretraining_modules()
        if self.freeze_hubert:
            for param in self.hubert_model.parameters():
                param.requires_grad = False
        
        # speech subsample
        if args.conv_kernel_sizes:
            self.subsampler = Conv1dSubsampler(
                hubert_args.model.encoder_embed_dim,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
        else:
            self.subsampler = None
            self.dim_proj = nn.Linear(hubert_args.model.encoder_embed_dim, args.encoder_embed_dim)
        
        # embedding
        self.embed_tokens = embed_tokens
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_tokens.embedding_dim, export=export)
        else:
            self.layernorm_embedding = None
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, 
            args.encoder_embed_dim,
            self.padding_idx,
        )

        # transformer encoder
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def _get_hubert_features(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        hubert_args = {
            "source": src_tokens,
            "padding_mask": padding_mask,
            "mask": False,
        }
        x, padding_mask = self.hubert_model.extract_features(**hubert_args)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return x, padding_mask, output_length
    
    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def _forward(self, src_tokens, src_lengths, mode, return_all_hiddens=False):
        if mode == "st":
            x, encoder_padding_mask, input_lengths = self._get_hubert_features(src_tokens, src_lengths)
            if self.subsampler is not None:
                x, input_lengths = self.subsampler(x, input_lengths)
                encoder_padding_mask = lengths_to_padding_mask(input_lengths)
                x = x.transpose(0, 1)  # T x B x C -> B x T x C
            else:
                x = self.dim_proj(x)
            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)
            x = self.dropout_module(x)
        else:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
            x, _ = self.forward_embedding(src_tokens)
            if has_pads:
                x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        encoder_embedding = x
        x = x.transpose(0, 1)  # B x T x C -> T x B x C

        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths, mode, return_all_hiddens=False):
        x = self._forward(
            src_tokens, src_lengths, mode, return_all_hiddens=return_all_hiddens
        )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

@register_model_architecture(model_name="hubert_transformer", arch_name="hubert_transformer")
def base_architecture(args):
    # subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

@register_model_architecture(model_name="hubert_transformer", arch_name="hubert_transformer_postln")
def hubert_transformer_postln(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    base_architecture(args)