import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PreTrainedModel

from contrastors.distributed import gather_with_grad
from contrastors.models.biencoder import BiEncoder, LogitScale


class DualEncoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vision = BiEncoder(config.image_model_args)
        text_model_args = config.text_model_args
        if text_model_args.precomputed:
            assert text_model_args.freeze, "Precomputed text model must be frozen"
            self.precomputed_text = True
        else:
            self.precomputed_text = False

        self.text = BiEncoder(config.text_model_args)

        self.logit_scale = LogitScale(config.image_model_args)

    def encode_text(self, text, normalize=True):
        text_outputs = self.text(**text, normalize=normalize)

        return text_outputs["embedding"]

    def encode_image(self, vision, normalize=True):
        vision_outputs = self.vision(vision, normalize=normalize)

        return vision_outputs["embedding"]

    def forward(self, text_inputs, vision_inputs):
        if self.precomputed_text:
            assert "text_embs" in text_inputs, "Precomputed text inputs must have text_embs"

            text_outputs = {"embedding": text_inputs["text_embs"]}
            vision_outputs = self.vision(**vision_inputs, normalize=False)
        else:
            text_outputs = self.text(**text_inputs, normalize=False)
            vision_outputs = self.vision(**vision_inputs, normalize=False)

        metrics = {}

        text_emb = F.normalize(text_outputs["embedding"], dim=-1, p=2)
        all_text_emb = gather_with_grad(text_emb)

        vision_emb = F.normalize(vision_outputs["embedding"], dim=-1, p=2)
        all_vis_emb = gather_with_grad(vision_emb)

        logits_per_image = self.logit_scale(vision_emb @ all_text_emb.T)
        logits_per_text = self.logit_scale(text_emb @ all_vis_emb.T)

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits).to(logits_per_image.device)
        labels = labels + num_logits * dist.get_rank()

        image_text_loss = (
            (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels))
            / 2
            * dist.get_world_size()
        )
        metrics = {"loss": image_text_loss, "image_text_loss": image_text_loss}

        return metrics
