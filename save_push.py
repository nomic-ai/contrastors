from contrastors.models.biencoder import BiEncoder, BiEncoderConfig
from contrastors.models.huggingface.configuration_hf_nomic_bert import NomicBertConfig
from contrastors.models.huggingface.modeling_hf_nomic_bert import NomicBertModel

config = BiEncoderConfig.from_pretrained(
    "/home/paperspace/contrastors-dev/src/contrastors/ckpts/matryoshka-normed-correctly/epoch_0_model"
)
model = BiEncoder.from_pretrained(
    "/home/paperspace/contrastors-dev/src/contrastors/ckpts/matryoshka-normed-correctly-equal-weights/epoch_0_model", config=config
)
import pdb; pdb.set_trace()


NomicBertConfig.register_for_auto_class()
NomicBertModel.register_for_auto_class("AutoModel")

model = model.trunk
# model = NomicBertModel.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
model.push_to_hub("nomic-ai/nomic-embed-text-v1p5-equal", private=True)
