import os
from functools import partial

import torch
from safetensors.torch import load_file as safe_load_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files


# adapted from flash attention, added safe serialization option for hf models
def state_dict_from_pretrained(model_name, safe_serialization=False, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    is_sharded = False
    load_safe = False
    resolved_archive_file = None

    weights_path = os.path.join(model_name, WEIGHTS_NAME)
    weights_index_path = os.path.join(model_name, WEIGHTS_INDEX_NAME)
    safe_weights_path = os.path.join(model_name, SAFE_WEIGHTS_NAME)
    safe_weights_index_path = os.path.join(model_name, SAFE_WEIGHTS_INDEX_NAME)

    if os.path.isfile(weights_path):
        resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    elif os.path.isfile(weights_index_path):
        resolved_archive_file = cached_file(model_name, WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False)
        is_sharded = True
    elif os.path.isfile(safe_weights_path):
        resolved_archive_file = cached_file(model_name, SAFE_WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
        load_safe = True
    elif os.path.isfile(safe_weights_index_path):
        resolved_archive_file = cached_file(
            model_name, SAFE_WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False
        )
        is_sharded = True
        load_safe = True
    else:  # Try loading from HF hub instead of from local files
        weight_name = WEIGHTS_NAME if not safe_serialization else SAFE_WEIGHTS_NAME
        resolved_archive_file = cached_file(
            model_name, weight_name, token=True, _raise_exceptions_for_missing_entries=False
        )
        if resolved_archive_file is None:
            weight_index = WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
            resolved_archive_file = cached_file(
                model_name, weight_index, token=True, _raise_exceptions_for_missing_entries=False
            )
            if resolved_archive_file is not None:
                is_sharded = True

        load_safe = safe_serialization

    if resolved_archive_file is None:
        raise EnvironmentError(f"Model name {model_name} was not found.")

    if load_safe:
        loader = partial(safe_load_file, device=mapped_device)
    else:
        loader = partial(torch.load, map_location=mapped_device)

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different
        # checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(model_name, resolved_archive_file)
        state_dict = {}
        for sharded_file in resolved_archive_file:
            state_dict.update(loader(sharded_file))
    else:
        state_dict = loader(resolved_archive_file)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict


def filter_shapes(state_dict, model):
    """
    Filters the state dict to match the current model shape.
    """
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model.state_dict():
            if value.shape == model.state_dict()[key].shape:
                filtered_state_dict[key] = value
    return filtered_state_dict
