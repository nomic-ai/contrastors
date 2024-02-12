import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from contrastors.models.decoder import DecoderModel
from contrastors.models.decoder.gpt_neox import gpt_neox_config_to_gpt2_config
from contrastors.models.decoder.open_lm import open_lm_config_to_gpt2_config
from contrastors.models.encoder import NomicBertModel


class LogitScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(config.logit_scale), requires_grad=config.trainable_logit_scale
        )

    def forward(self, x):
        return x * self.logit_scale.exp()


class ClsSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_ids, attention_mask):
        return hidden_states[:, 0]


class LastTokenPooling(nn.Module):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id

    def forward(self, hidden_states, input_ids, attention_mask):
        # get the embedding corresponding to the first eos token
        # we don't substract 1 because the eos token is already included in the input_ids and attention_mask
        # and we want to get the embedding of the last token
        sequence_lengths = attention_mask.sum(-1) - 1
        selected_tokens = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths]

        if not torch.all(selected_tokens == self.eos_token_id):
            raise ValueError(
                f"The last token of the input_ids is not the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
            )
        prev_token = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths - 1]
        if torch.any(prev_token == self.eos_token_id):
            raise ValueError(
                f"The second to last token of the input_ids is the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
            )

        embs = hidden_states[torch.arange(hidden_states.shape[0]), sequence_lengths]

        return embs


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_ids, attention_mask):
        s = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(axis=1, keepdim=True).float()
        return s / d


class BiEncoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.use_fused_kernels:
            if config.encoder is False:
                model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
                if "gpt-neox" in config.model_name or "pythia" in config.model_name:
                    model_config = gpt_neox_config_to_gpt2_config(model_config)
                elif "open_lm" in config.model_name:
                    model_config = open_lm_config_to_gpt2_config(model_config)
                self.trunk = DecoderModel.from_pretrained(
                    config.model_name, config=model_config, safe_serialization=True
                )
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self.eos_token_id = tokenizer.eos_token_id
            else:
                self.trunk = NomicBertModel.from_pretrained(
                    config.model_name,
                    add_pooling_layer=False,
                    rotary_scaling_factor=getattr(config, "rotary_scaling_factor", None),
                )
        else:
            self.trunk = AutoModel.from_pretrained(config.model_name, trust_remote_code=True, add_pooling_layer=False)

        if config.freeze:
            self.trunk.eval()
            for param in self.trunk.parameters():
                param.requires_grad = False

        if config.projection_dim:
            self.proj = nn.Linear(self.trunk.config.hidden_size, config.projection_dim)
        else:
            self.proj = nn.Identity()

        if config.pooling == "mean":
            self.selector = MeanPooling()
        elif config.pooling == "last":
            self.selector = LastTokenPooling(self.eos_token_id)
        elif config.pooling == "cls":
            self.selector = ClsSelector()
        else:
            raise ValueError(f"Pooling {config.pooling} not supported")

        if config.hamming:
            self.hamming = nn.LayerNorm(self.trunk.config.hidden_size, elementwise_affine=False)
        else:
            self.hamming = nn.Identity()

    def forward(self, input_ids, attention_mask=None, is_padded_inputs=True, normalize=True, binarize=False, **kwargs):
        trunk_output = self.trunk(input_ids, attention_mask=attention_mask, **kwargs)
        trunk_output = trunk_output[0]
        trunk_output = self.proj(trunk_output)
        embedding = self.selector(trunk_output, input_ids, attention_mask)
        embedding = self.hamming(embedding)
        if binarize:
            return {"embedding": (embedding > 0).float()}
        elif normalize:
            return {"embedding": F.normalize(embedding, dim=-1)}
        else:
            return {"embedding": embedding}
