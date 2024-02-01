# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

import logging
import os
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm, rms_norm
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import GPT2Config, PreTrainedModel
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertForPreTrainingOutput,
    SequenceClassifierOutput,
)

from contrastors.layers import Block
from contrastors.layers.embedding import BertEmbeddings
from contrastors.models.encoder.bert import remap_bert_state_dict
from contrastors.models.model_utils import state_dict_from_pretrained

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm, layer_norm
except ImportError:
    dropout_add_layer_norm, layer_norm = None, None

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None


logger = logging.getLogger(__name__)


class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    config_class = GPT2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config)
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPretraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # TODO: fix this
        # Assuming we know what we're doing when loading from disk
        # Prob a bad assumption but i'm tired and want to train this asap
        if os.path.exists(model_name):
            load_return = model.load_state_dict(torch.load(f"{model_name}/pytorch_model.bin"), strict=False)
        else:
            load_return = model.load_state_dict(
                remap_bert_state_dict(state_dict_from_pretrained(model_name), config), strict=True
            )
        logger.info(load_return)
        return model

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class BertEncoder(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_padded_inputs: Optional[bool] = True,
    ):
        """If subset_mask is not None, we only want output for the subset of the sequence.
        This means that we only compute the last layer output for these tokens.
        subset_mask: (batch, seqlen), dtype=torch.bool
        """
        hidden_states2 = None
        residual = None

        batch, seqlen = hidden_states.shape[:2]
        if self.config.rotary_emb_fraction <= 0:
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask)
        else:
            indices = None
            cu_seqlens = None
            max_seqlen_in_batch = None
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                hidden_states, hidden_states2, residual = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    hidden_states2,
                    residual,
                    attention_mask,
                    None,
                    None,
                    is_padded_inputs,
                    # if you freeze ANY layers, you need `use_reentrant=False`
                    # https://github.com/huggingface/transformers/issues/21381
                    # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/7
                    use_reentrant=False,
                )

            else:
                hidden_states, hidden_states2, residual = layer(
                    hidden_states,
                    hidden_states2,
                    residual,
                    attention_mask,
                    position_ids,
                    None,
                    is_padded_inputs,
                    output_attentions,
                    use_cache,
                    cu_seqlens=cu_seqlens,
                    max_seq_len=max_seqlen_in_batch,
                )
        if self.config.rotary_emb_fraction <= 0:
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.n_embd, config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.n_embd, config.n_embd, bias=config.mlp_fc1_bias)
        approximate = "tanh" if config.activation_function in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"] else "none"
        if config.activation_function == "swiglu":
            self.transform_act_fn = F.silu
        else:
            self.transform_act_fn = nn.GELU(approximate=approximate)
        norm_cls = partial(
            nn.LayerNorm if not config.use_rms_norm else RMSNorm,
            eps=config.layer_norm_epsilon,
        )
        self.layer_norm = norm_cls(config.n_embd)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.layer_norm(hidden_states)
        else:
            if isinstance(self.layer_norm, RMSNorm):
                hidden_states = rms_norm(hidden_states, self.layer_norm.weight, self.layer_norm.eps)
            elif isinstance(self.layer_norm, nn.LayerNorm):
                hidden_states = layer_norm(
                    hidden_states, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps
                )
            else:
                raise ValueError(f"Unsupported layer norm class: {self.layer_norm.__class__.__name__}")

        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense

        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # TODO: assumes that mlp_fc1_bias == mlp_fc2_bias, we should enforce that somewhere
        self.decoder = linear_cls(config.n_embd, config.vocab_size, bias=config.mlp_fc1_bias)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertModel(BertPreTrainedModel):
    def __init__(self, config: GPT2Config, add_pooling_layer=True):
        super().__init__(config)
        self.pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        if config.vocab_size % self.pad_vocab_size_multiple != 0:
            config.vocab_size += self.pad_vocab_size_multiple - (config.vocab_size % self.pad_vocab_size_multiple)
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_pytorch_tanh",
            "swiglu",
            "geglu",
            "glu",
        ]

        self.embeddings = BertEmbeddings(config)
        self.emb_drop = nn.Dropout(config.resid_pdrop)
        self.emb_ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_tokens_mask=None,
    ):
        """If masked_tokens_mask is not None (i.e. last_layer_subset == True in BertForPreTraining),
        we only want the output for the masked tokens. This means that we only compute the last
        layer output for these tokens.
        masked_tokens_mask: (batch, seqlen), dtype=torch.bool
        """
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        # TD [2022-12:18]: Don't need to force residual in fp32
        # BERT puts embedding LayerNorm before embedding dropout.
        if not self.fused_dropout_add_ln:
            hidden_states = self.emb_ln(hidden_states)
        else:
            hidden_states = layer_norm(hidden_states, self.emb_ln.weight, self.emb_ln.bias, self.emb_ln.eps)
        hidden_states = self.emb_drop(hidden_states)

        if masked_tokens_mask is not None:
            batch_size, seqlen = input_ids.shape[:2]
            # We also need the first column for the CLS token
            first_col_mask = torch.zeros(batch_size, seqlen, dtype=torch.bool, device=input_ids.device)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask
        else:
            subset_mask = None

        sequence_output = self.encoder(hidden_states, attention_mask=attention_mask)

        if masked_tokens_mask is None:
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            if attention_mask is not None:
                subset_idx = subset_mask[attention_mask]
                pool_input = sequence_output[first_col_mask[attention_mask][subset_idx]]
                sequence_output = sequence_output[masked_tokens_mask[attention_mask][subset_idx]]
            else:
                pool_input = sequence_output[first_col_mask[subset_mask]]
                sequence_output = sequence_output[masked_tokens_mask[subset_mask]]
            pooled_output = self.pooler(pool_input, pool=False) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class BertForPreTraining(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        # If dense_seq_output, we only need to pass the hidden states for the masked out tokens
        # (around 15%) to the classifier heads.
        self.dense_seq_output = getattr(config, "dense_seq_output", False)
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, "last_layer_subset", False)
        if self.last_layer_subset:
            assert self.dense_seq_output, "last_layer_subset requires dense_seq_output"
        use_xentropy = getattr(config, "use_xentropy", False)
        if use_xentropy and CrossEntropyLoss is None:
            raise ImportError("xentropy_cuda is not installed")
        loss_cls = nn.CrossEntropyLoss if not use_xentropy else partial(CrossEntropyLoss, inplace_backward=True)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertPreTrainingHeads(config)
        self.mlm_loss = loss_cls()

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        """
        If labels are provided, they must be -100 for masked out tokens (as specified in the attention
        mask).
        Outputs:
            if `labels` and `next_sentence_label` are not `None`:
                Outputs the total_loss which is the sum of the masked language modeling loss and the next
                sentence classification loss.
            if `labels` or `next_sentence_label` is `None`:
                Outputs a tuple comprising
                - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
                - the next sentence classification logits of shape [batch_size, 2].

        """
        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            masked_tokens_mask=masked_tokens_mask,
        )
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        if self.dense_seq_output and labels is not None:
            masked_token_idx = torch.nonzero(labels.flatten() >= 0, as_tuple=False).flatten()
            if not self.last_layer_subset:
                sequence_output = index_first_axis(rearrange(sequence_output, "b s d -> (b s) d"), masked_token_idx)
        prediction_scores = self.cls(sequence_output)

        total_loss = None
        if labels is not None:
            if self.dense_seq_output and labels is not None:  # prediction_scores are already flattened
                masked_lm_loss = self.mlm_loss(prediction_scores, labels.flatten()[masked_token_idx])
            else:
                masked_lm_loss = self.mlm_loss(
                    rearrange(prediction_scores, "... v -> (...) v"),
                    rearrange(labels, "... -> (...)"),
                )
            total_loss = masked_lm_loss.float()

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
        )


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.dense_seq_output = getattr(config, "dense_seq_output", False)
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, "last_layer_subset", False)
        if self.last_layer_subset:
            assert self.dense_seq_output, "last_layer_subset requires dense_seq_output"

        self.bert = BertModel(config)
        classifier_dropout = getattr(config, "classifier_dropout", config.embd_pdrop)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.n_embd, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            masked_tokens_mask=masked_tokens_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
