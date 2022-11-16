# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
from sklearn.metrics import mean_squared_error, r2_score
import fairseq.tasks.sentence_prediction
import fairseq.tasks.masked_lm
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.models import FairseqEncoder
from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
from fairseq.data import (MaskTokensDataset,
                          LanguagePairDataset,
                          PrependTokenDataset,
                          data_utils)
from fairseq.models import register_model, register_model_architecture, BaseFairseqModel
from fairseq.models.roberta import TransformerSentenceEncoder, RobertaEncoder, RobertaModel
from musicbert.roberta.model import RobertaRegressionHead
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from functools import lru_cache
from typing import Optional, Tuple
import numpy as np
import math
import logging
import os
import json
from fairseq import utils
from fairseq.models import BaseFairseqModel
import torch
from torch import Tensor
from typing import Union, Callable
from itertools import count
from .LASCon import LASCon

def exists(value):
    return value is not None

def default(value, default):
    if exists(value):
        return value
    return default

def kl_loss(input, target, reduction="batchmean"):
    return F.kl_div(
        input = F.logsigmoid(input), 
        target =target,
        reduction=reduction,
    )


logger = logging.getLogger(__name__)
disable_cp = 'disable_cp' in os.environ
print('disable_cp =', disable_cp)
mask_strategy = os.environ['mask_strategy'].split(
    '+') if 'mask_strategy' in os.environ else ['bar']
print('mask_strategy =', mask_strategy)
assert all(item in ['element', 'compound', 'bar'] for item in mask_strategy)
convert_encoding = os.environ['convert_encoding'] if 'convert_encoding' in os.environ else 'OCTMIDI'
print('convert_encoding =', convert_encoding)
crop_length = int(os.environ['crop_length']
                  ) if 'crop_length' in os.environ else None
print('crop_length =', crop_length)  # of compound tokens
max_bars = 256
max_instruments = 256

# Thank GitHub user @neelansh for providing multi-label classification solution
# See https://github.com/pytorch/fairseq/issues/2169
@register_task("xai")
class MusicBERTSentencePredictionMultilabelTaskXAI(SentencePredictionTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--num-cls-classes",
            type=int,
            default=-1,
            help="number of class targets",
        )
        parser.add_argument(
            "--num-reg-classes",
            type=int,
            default=-1,
            help="number of regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--regression-target", action="store_true", default=False) #don't use
        parser.add_argument("--no-shuffle", action="store_true", default=False) #don't use
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )

    def load_dataset(self, split, combine=False, **kwargs):
        split_path = os.path.join(self.args.data, 'input0', split)
        input0 = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if self.args.init_token is not None:
            input0 = OctupleTokenDataset(input0)
        src_dataset = input0
        labels, label_lengths = [], []
        with open(os.path.join(self.args.data, 'label', split+".label")) as file:
            for line in file:
                line = line.strip()
                label = json.loads(line)
                label = torch.tensor(label)
                labels.append(label)
                label_lengths.append(len(label))
                #assert len(label) == self.args.num_reg_classes + 1, print(len(label), self.args.num_reg_classes)
        assert len(src_dataset) == len(labels)
        self.datasets[split] = LanguagePairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.label_dictionary,
            tgt=labels,
            tgt_sizes=torch.tensor(label_lengths),
            tgt_dict=self.label_dictionary,
            left_pad_source=False,
            input_feeding=False,
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_cls_classes > 0, "Must set --num-cls-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, "label", "dict.txt"),
                source=False,
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(args, data_dict, label_dict)

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_cls_classes,
        )
        if self.args.num_reg_classes > 1:
            model.register_regression_head(
                getattr(args, "regression_head_name", "sentence_regression_head"),
                num_classes=self.args.num_reg_classes,
            )

        return model

@register_criterion("M2P_xai")
class MusicBERTM2PCriterionForXAI(SentencePredictionCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits])
        #sample_size = targets.numel()
        sample_size = logits.size()[0]

        targets = targets[:,-1]
        loss_fct = nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(logits, targets.long())

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        preds = logits.argmax(dim=1)
        logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False

class OctupleMaskTokensDataset(MaskTokensDataset):
    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)
            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )
            assert not self.mask_whole_words, 'mask whole words not supported for cp'

            def generate_mask(sz, prob):
                mask_n = np.random.rand(sz)
                mask_s = np.zeros(sz, dtype=np.int8)
                mask_s += mask_n < prob * \
                    (self.random_token_prob)  # 3 -> random
                mask_s += mask_n < prob * \
                    (self.random_token_prob +
                     self.leave_unmasked_prob)  # 2 -> original
                mask_s += mask_n < prob * 1.00  # 1 -> [mask]
                return mask_s
            mask_prob = self.mask_prob
            mask = np.zeros_like(item, dtype=np.int8)
            # mask bos eos tokens (compound)
            mask[:8] = np.repeat(generate_mask(1, mask_prob), 8)
            # mask bos eos tokens (compound)
            mask[-8:] = np.repeat(generate_mask(1, mask_prob), 8)
            strategy = np.random.choice(mask_strategy)
            if strategy == 'element':  # element level mask
                mask[8: -8] = np.repeat(generate_mask(sz -
                                                      2 * 8, mask_prob), 1)
            if strategy == 'compound':  # compound token level mask
                mask[8: -8] = np.repeat(generate_mask(sz //
                                                      8 - 2, mask_prob), 8)
            if strategy == 'bar':  # bar level mask
                mask[8: -8] = generate_mask((max_bars * max_instruments + len(self.vocab)) * 8, mask_prob).reshape(-1, 8)[
                    ((item[8: -8: 8] - 4) * max_instruments) + (item[8 + 2: -8 + 2: 8] - 4)].flatten()
            if self.return_masked_tokens:
                new_item = item.numpy()[:]
                new_item[mask == 0] = self.pad_idx
                return torch.from_numpy(new_item)
            masked_item = np.random.choice(len(self.vocab), sz)
            set_original = np.isin(mask, [0, 2])
            masked_item[set_original] = item[set_original]
            set_mask = np.isin(mask, [1])
            masked_item[set_mask] = self.mask_idx
            return torch.from_numpy(masked_item)


class OctupleEncoder(TransformerSentenceEncoder):
    def __init__(self, *args, **kwargs) -> None:
        self.adv_training = kwargs.pop('adv_training')
        super().__init__(*args, **kwargs)
        self.tpu = False
        embedding_dim = kwargs['embedding_dim']
        if not disable_cp:
            self.downsampling = nn.Sequential(
                nn.Linear(embedding_dim * 8, embedding_dim))
            self.upsampling = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 8))
        self.attn_mask = None
        self.num_attention_heads = kwargs['num_attention_heads']

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None, # None 
        last_state_only: bool = False, # True
        positions: Optional[torch.Tensor] = None, # None
        token_embeddings: Optional[torch.Tensor] = None, # None으로 들어옴
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ratio = 1 if disable_cp else 8
        if not disable_cp: #disable_cp=False
            assert tokens.shape[1] % ratio == 0, 'token sequences length should be multiple of ' + str(
                ratio) + ' for compound mode'
            assert last_state_only, 'hidden states not available for compound mode'
            assert positions is None, 'custom positions is not supported for compound mode'
            #assert token_embeddings is None, 'custom token embeddings is not supported for compound mode'
            assert segment_labels is None, 'segment embedding not supported for compound mode'
        padding_mask = tokens[:, ::ratio].eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None
        if token_embeddings is not None:
            x = token_embeddings
            #print('use custom token embedding')
        else:
            x = self.embed_tokens(tokens)
        if not disable_cp:
            x = self.downsampling(x.view(x.shape[0], x.shape[1] // ratio, -1))
        if self.embed_scale is not None:
            x = x * self.embed_scale
        if self.embed_positions is not None:
            x = x + \
                self.embed_positions(tokens[:, ::ratio], positions=positions)
        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)
        
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        # why transpose?
        x = x.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            #x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask = self.attn_mask)[0]
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)
        if not disable_cp:
            x = x.transpose(0, 1)
            x = self.upsampling(x).view(x.shape[0], x.shape[1] * ratio, -1)
            x = x.transpose(0, 1)
        sentence_rep = x[0, :, :]
        if last_state_only:
            inner_states = [x]
        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            #print(len(inner_states), inner_states[0].shape)
            #print(sentence_rep.shape)
            return inner_states, sentence_rep

class MusicBERTEncoder(RobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.sentence_encoder = OctupleEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            adv_training = args.adv,
        )
    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        token_embeddings = None,
        **unused,
    ):
        #print("unused:", kwargs)
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens, token_embeddings = token_embeddings
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra        

    def extract_features(self, src_tokens, return_all_hiddens=False, token_embeddings=None):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
            token_embeddings=token_embeddings,
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {"inner_states": inner_states if return_all_hiddens else None}


@register_model("musicbert")
class MusicBERTModel(RobertaModel):

    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        self.regression_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
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
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weight s-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        parser.add_argument(
            "--spectral-norm-regression-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the regression head",
        )
        
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        encoder = MusicBERTEncoder(args, task.source_dictionary)
        return cls(args, encoder)
    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        regression_head_name = None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True
        #print("musicbertmodelforawrd", kwargs.keys())
        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, token_embeddings = kwargs.get("token_embeddings", None))
        if classification_head_name is not None:
            x1 = self.classification_heads[classification_head_name](x)
            if regression_head_name is not None: #M2PFnP
                x2 = self.regression_heads[regression_head_name](x)
                return (x1, x2), extra
            else:
                return x1, extra
        else:
            return x, extra

    def register_regression_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a regression head."""
        if name in self.regression_heads:
            prev_num_classes = self.regression_heads[name].out_proj.out_features
            prev_inner_dim = self.regression_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        # can be changed to custom regression head
        self.regression_heads[name] = RobertaRegressionHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_regression_head,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        print(state_dict.keys())
        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


        # Handle new regression heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "regression_heads")
            else self.regression_heads.keys()
        )
        #print(current_head_names)
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "regression_heads."):
                continue
            
            head_name = k[len(prefix + "regression_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "regression_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "regression_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_regression_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting regression head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.regression_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.regression_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting regression head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added regression heads into the state dict
        # with their current weights.
        if hasattr(self, "regression_heads"):
            cur_state = self.regression_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "regression_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "regression_heads." + k)
                    state_dict[prefix + "regression_heads." + k] = v

@register_model_architecture("musicbert", "musicbert")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )
    args.spectral_norm_regression_head = getattr(
        args, "spectral_norm_regression_head", False
    )
    args.adv = getattr(args, "adv", False)


@register_model_architecture("musicbert", "musicbert_base")
def musicbert_base_architecture(args):
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_large")
def musicbert_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_medium")
def musicbert_medium_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_small")
def musicbert_small_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_mini")
def musicbert_mini_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_tiny")
def musicbert_tiny_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    base_architecture(args)


class OctupleTokenDataset(PrependTokenDataset):
    def adaptor(self, e):
        prev_bar = None
        prev_pos = None
        prev_prog = None
        new_e = []
        for i in e:
            if prev_bar != i[0]:
                prev_bar = i[0]
                prev_pos = None
                new_e.append((i[0], None, None, None, None, None, i[6], None))
            if prev_pos != i[1]:
                prev_pos = i[1]
                prev_prog = None
                new_e.append((None, i[1], None, None, None, None, None, i[7]))
            if prev_prog != i[2]:
                prev_prog = i[2]
                new_e.append((None, None, i[2], None, None, None, None, None))
            if True:
                new_e.append((None, None, None, i[3], i[4], i[5], None, None))
        return new_e

    def convert(self, item):
        encoding = item[8: -8].tolist()
        encoding = list(tuple(encoding[i: i + 8])
                        for i in range(0, len(encoding), 8))
        encoding = self.adaptor(encoding)
        if convert_encoding == 'CP':
            encoding = list(3 if j is None else j for i in encoding for j in i)[
                :crop_length * 8]
        elif convert_encoding == 'REMI':
            encoding = list(j for i in encoding for j in i if j is not None)[
                :crop_length]
        else:
            assert False, 'Unknown encoding format'
        bos = 0
        eos = 2
        encoding = ([bos] * 8) + encoding + ([eos] * 8)
        return torch.tensor(encoding)

    def __init__(self, dataset, token=None):
        super().__init__(dataset, token=None)
        if convert_encoding != 'OCTMIDI':
            self._sizes = np.array([len(self.convert(i)) for i in dataset])
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if convert_encoding != 'OCTMIDI':
            item = self.convert(item)
        return item

    def num_tokens(self, index):
        return self._sizes[index].item()

    def size(self, index):
        return self._sizes[index].item()



class AnnotationEncoder(nn.Module):
    def __init__(self, in_dim=26, out_dim = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.GELU(), 
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

@register_model("xai_pretrain_model")
class XAIPretrainModel(BaseFairseqModel):
    def __init__(self, mainEncoder, annotationEncoder, classificationHead, projectionHead, is_fine_tuning):
        super().__init__()
        self.mainEncoder = mainEncoder
        self.annotationEncoder = annotationEncoder
        self.classificationHead = classificationHead
        self.projectionHead = projectionHead
        self.is_fine_tuning = is_fine_tuning

    @staticmethod
    def add_args(parser):
        MusicBERTModel.add_args(parser)
        parser.add_argument("--is-fine-tuning", action='store_true', default=False)

    @classmethod
    def build_model(cls, args, task):
        mainEncoder = MusicBERTModel.build_model(args, task)
        annotationEncoder = AnnotationEncoder()
        classificationHead = AnnotationEncoder(in_dim=768, out_dim=14)
        projectionHead = AnnotationEncoder(in_dim=768, out_dim=768)
        return cls(mainEncoder, annotationEncoder, classificationHead, projectionHead, args.is_fine_tuning)

    def forward(self, src_tokens, annotation=None, **kwargs):
        if self.is_fine_tuning:
            x, extra = self.mainEncoder(src_tokens, features_only=True)
            x = self.classificationHead(x[:, 0])
            return x, extra
        else:
            x1, extra = self.mainEncoder(src_tokens, features_only=True)
            x1 = x1[:, 0]
            x2 = self.annotationEncoder(annotation)
            return x1, x2, extra 
    
    def upgrade_state_dict(self, state_dict):
        if 'encoder.sentence_encoder.downsampling.0.weight' in list(state_dict.keys()):
            ke = list(state_dict.keys())
            for k in ke:
                state_dict['mainEncoder.' + k] = state_dict[k]
                del state_dict[k]
            s = self.annotationEncoder.state_dict()
            for k in s.keys():
                state_dict['annotationEncoder.' + k] = s[k]
            
            s = self.classificationHead.state_dict()
            for k in s.keys():
                state_dict['classificationHead.' + k] = s[k]

            s = self.projectionHead.state_dict()
            for k in s.keys():
                state_dict['projectionHead.' + k] = s[k]


@register_model_architecture("xai_pretrain_model", "xai_pretrain_arch")
def xai_base_architecutre(args):
    base_architecture(args)
    
@register_task("xai_pretrain_task")
class MusicBERTXAIPretrainTask(MusicBERTSentencePredictionMultilabelTaskXAI):
    def load_dataset(self, split, combine=False, **kwargs):
        split_path = os.path.join(self.args.data, 'input0', split)
        input0 = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if self.args.init_token is not None:
            input0 = OctupleTokenDataset(input0)
        src_dataset = input0
        labels, label_lengths = [], []
        with open(os.path.join(self.args.data, 'label', split+".label")) as file:
            with open(os.path.join(self.args.data, 'label', split+".id")) as id_file:
                for line, id_line in zip(file, id_file):
                    line = line.strip()
                    id_line = id_line.strip()
                    performer = id_line.split('_')[4]
                    performer = 13 if performer == 'Score' else int(performer)
                    performer = int(performer)
                    performer = torch.tensor([performer])
                    label = json.loads(line)
                    label = torch.tensor(label)
                    label = torch.cat([performer, label])
                    labels.append(label)
                    label_lengths.append(len(label))
                    #assert len(label) == self.args.num_reg_classes + 1, print(len(label), self.args.num_reg_classes)
        assert len(src_dataset) == len(labels)
        self.datasets[split] = LanguagePairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.label_dictionary,
            tgt=labels,
            tgt_sizes=torch.tensor(label_lengths),
            tgt_dict=self.label_dictionary,
            left_pad_source=False,
            input_feeding=False,
        )

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)
        return model

@register_criterion("xai_pretrain_loss")
class MusicBERTM2PCriterionForXAIPretrain(MusicBERTM2PCriterionForXAI):
    label_sim='supcon'
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        embedding1, embedding2, _ = model(
            **sample["net_input"],
            annotation = sample['target'][:,1:],
            features_only=True,
        )
        targets = sample['target'][:, 0]
        sample_size = targets.shape[0]

        embedding1 = torch.nn.functional.normalize(embedding1)
        embedding2 = torch.nn.functional.normalize(embedding2)

        loss_fct = LASCon(label_sim=self.label_sim)
        loss = loss_fct(embedding1, embedding2, targets, targets)

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
            "minus_loss": -(loss.data)
        }

        return loss, sample_size, logging_output

@register_criterion("xai_pretrain_loss_unimodal")
class MusicBERTM2PCriterionForXAIPretrainUnimodal(MusicBERTM2PCriterionForXAI):
    label_sim='supcon'
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        embedding1, _, _ = model(
            **sample["net_input"],
            annotation = sample['target'][:,1:],
            features_only=True,
        )
        targets = sample['target'][:, 0]
        sample_size = targets.shape[0]
        half_sample_size = sample_size//2

        embedding1 = torch.nn.functional.normalize(embedding1)

        loss_fct = LASCon(label_sim =self.label_sim)
        loss = loss_fct(embedding1[0:half_sample_size], embedding1[half_sample_size:], targets[0:half_sample_size], targets[half_sample_size:])

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
            "minus_loss": -(loss.data)
        }

        return loss, sample_size, logging_output

@register_criterion("xai_pretrain_loss_ntxent")
class MusicBERTM2PCriterionForXAIPretrainNTXent(MusicBERTM2PCriterionForXAIPretrain):
    label_sim='nt-xent'

@register_criterion("xai_pretrain_loss_unimodal_ntxent")
class MusicBERTM2PCriterionForXAIPretrainUnimodalNTXent(MusicBERTM2PCriterionForXAIPretrainUnimodal):
    label_sim='nt-xent'

@register_criterion("xai_finetuning_loss")
class MusicBERTM2PCriterionForXAIFinetuning(SentencePredictionCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        pred, _, = model(
            **sample["net_input"],
            features_only=True
        )
        targets = sample['target'][:, 0]

        sample_size = pred.size()[0]
        loss_fct = nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(pred, targets.long())

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        preds = pred.argmax(dim=1)
        logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output

fairseq.tasks.sentence_prediction.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.masked_lm.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.masked_lm.MaskTokensDataset = OctupleMaskTokensDataset
