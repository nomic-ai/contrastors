import html
import re

import ftfy
import torch
import torch.nn.functional as F
from tqdm import tqdm

# this is from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
EMERGENT_ZS_TEMPLATE = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def zeroshot_classifier_weights(
    model,
    tokenizer,
    device,
    classnames,
    templates=None,
    dtype=torch.float32,
    return_text_embeddings=False,
    add_eos=False,
    prefix=None,
):
    if templates is None:
        templates = EMERGENT_ZS_TEMPLATE

    text_embeddings = []

    with torch.no_grad():
        zeroshot_weights = []
        for c in tqdm(classnames):
            texts = [template.format(c) for template in templates]
            texts = [whitespace_clean(basic_clean(text)) for text in texts]
            if add_eos:
                texts = [text + tokenizer.eos_token for text in texts]

            if prefix:
                texts = [f"{prefix}: {text}" for text in texts]

            tokenized_text = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True).to(device)

            class_embeddings = model(**tokenized_text)["embedding"]
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            # average over templates
            class_embeddings = class_embeddings.mean(dim=0)
            # norm avg template class_embeddings
            class_embeddings /= class_embeddings.norm()
            if return_text_embeddings:
                feats = {
                    "embedding": class_embeddings.detach().cpu().to(torch.float32).numpy(),
                    "class": c,
                    "input": c,
                    "modality": "text",
                }
                text_embeddings.append(feats)

            zeroshot_weights.append(class_embeddings)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    # shape should be (embed_dim, num_classes)
    zeroshot_weights = zeroshot_weights.to(dtype)

    return zeroshot_weights, text_embeddings
