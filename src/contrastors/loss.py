from contextlib import nullcontext

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

from contrastors.distributed import gather, gather_dict

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
except:
    te = None
    Format = None
    DelayedScaling = None

from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def clip_loss(
    query,
    document,
    logit_scale,
    step=None,
    gather_enabled=False,
    negatives=None,
    kd_scores=None,
    alpha=0.2,
    tokenizer=None,
    accelerator=None,
    inputs=None,
    dataset="",
    bidirectional=False,
):
    """Calculates the InfoNCE Loss for a batch of queries and documents.
    Inspired by: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py#L66

    Assumes that query.shape[0] <= document.shape[0]
    This will work for non-square matrices as well

    params:
        query: torch.Tensor of shape N x D
        document: torch.Tensor of shape M x D where M >= N
        temperature: torch.Tensor of shape 1

    returns:
        torch.Tensor of shape 1 corresponding to the loss
    """
    if gather_enabled:
        query = gather(query)
        document = gather(document)
        if negatives is not None:
            negatives = gather(negatives)

    device = query.device
    labels = torch.arange(query.shape[0]).to(device)

    if query.dtype != document.dtype:
        document = document.to(query.dtype)
    bs = query.shape[0]
    if negatives is not None:
        # negatives is of shape (bs*num_negatives, D)
        # we only want the negatives corresponding to the query
        # reshape and extract the negatives corresponding to the query
        # negatives should be of shape (bs, D) after this
        reshaped_negative = negatives.reshape(bs, -1, negatives.shape[-1])
        num_negatives = reshaped_negative.shape[1]

        # sim_query_doc is of shape (bs, bs + bs*num_negatives)
        sim_query_doc = torch.matmul(query, torch.cat([document, negatives], dim=0).T)
        similarity_query_document = sim_query_doc * logit_scale

        loss = F.cross_entropy(similarity_query_document, labels) * dist.get_world_size()

        if kd_scores is not None:
            kd_scores = kd_scores.to(device)
            kd_scores = gather(kd_scores)
            # ignore where -1
            # get positive scores and hard negative scores
            # use logit scaled scores !!!
            positives = similarity_query_document.diag()
            negatives = similarity_query_document[:, bs:]

            sim_qd_scores = torch.cat([positives.unsqueeze(1), negatives], dim=1)

            # masked log softmax similar to: https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
            # mask = (kd_scores != -1).to(device)
            # sim_qd_scores = sim_qd_scores + (mask.float() + 1e-45).log()
            # kd_scores = kd_scores + (mask.float() + 1e-45).log()
            log_scores = torch.log_softmax(sim_qd_scores, dim=1)
            log_labels = torch.log_softmax(kd_scores, dim=1)
            kl_loss = F.kl_div(log_scores, log_labels, reduction="batchmean", log_target=True)

            accelerator.log(
                {"kd_loss": kl_loss.detach().cpu().item(), "ce_loss": loss.detach().cpu().item()}, step=step
            )

            # sync to cuda to gather then detach
            inputs = gather_dict({k: v.to(device) for k, v in inputs.items()})
            inputs = {k: v.detach().cpu() for k, v in inputs.items()}

            queries = tokenizer.batch_decode(inputs["query_input_ids"], skip_special_tokens=True)
            documents = tokenizer.batch_decode(inputs["document_input_ids"], skip_special_tokens=True)
            negatives = tokenizer.batch_decode(inputs["negative_input_ids"], skip_special_tokens=True)
            softmax_pred_ce_scores = torch.log_softmax(sim_qd_scores, dim=1).exp().detach().cpu().numpy().tolist()
            softmax_true_ce_scores = torch.log_softmax(kd_scores, dim=1).exp().detach().cpu().numpy().tolist()

            data = {"query": queries, "document": documents}
            reshaped_negatives = []
            for i in range(len(queries)):
                reshaped_negatives.append(negatives[i * num_negatives : (i + 1) * num_negatives])

            # add predicted correct score and true correct score
            data["sofmax_pred_ce_score_correct"] = [pred[0] for pred in softmax_pred_ce_scores]
            data["sofmax_true_ce_score_correct"] = [true[0] for true in softmax_true_ce_scores]

            for neg in range(num_negatives):
                data[f"negative_{neg}"] = [negs[neg] for negs in reshaped_negatives]
                data[f"softmax_pred_ce_score_negative_{neg}"] = [pred[neg + 1] for pred in softmax_pred_ce_scores]
                data[f"softmax_true_ce_score_negative_{neg}"] = [true[neg + 1] for true in softmax_true_ce_scores]

            table = wandb.Table(dataframe=pd.DataFrame(data))
            accelerator.log({"cross_encoder_tabe": table}, step=step)

            loss = loss * alpha + kl_loss

    else:
        similarity_query_document = torch.matmul(query, document.T) * logit_scale
        labels = labels * (document.size(0) // query.size(0))
        if bidirectional:
            similarity_document_query = torch.matmul(document, query.T) * logit_scale
            loss = (
                F.cross_entropy(similarity_query_document, labels) + F.cross_entropy(similarity_document_query, labels)
            ) * dist.get_world_size()
        else:
            loss = F.cross_entropy(similarity_query_document, labels) * dist.get_world_size()

    if accelerator is not None:
        accuracy = (similarity_query_document.argmax(dim=1) == labels).float().mean()
        accelerator.log({f"accuracy_{dataset}": accuracy.detach().cpu().item()}, step=step)

    return loss


def gte_loss(query: torch.Tensor, document: torch.Tensor, logit_scale, gather_enabled=False):
    """Improved Contrastive Loss from https://arxiv.org/abs/2308.03281

    Calculates an improved contrastive loss by adding query to query similarity
    and document to document similarity to the original clip loss.

    params:
        query: torch.Tensor of shape N x D
        document: torch.Tensor of shape M x D where M >= N
        temperature: torch.Tensor of shape 1

    returns:
        torch.Tensor of shape 1 corresponding to the loss
    """
    device = query.device
    indices = torch.arange(query.shape[0]).to(device)

    if query.dtype != document.dtype:
        document = document.to(query.dtype)

    if gather_enabled:
        query = gather(query)
        document = gather(document)

    sim_q_d = torch.matmul(query, document.T)
    sim_q_q = torch.matmul(query, query.T)
    sim_d_d = torch.matmul(document, document.T)
    sim_d_q = sim_q_d.T

    sim_q_d_logit_scale = logit_scale(sim_q_d)
    sim_q_q_logit_scale = logit_scale(sim_q_q)
    sim_d_d_logit_scale = logit_scale(sim_d_d)
    max_val = torch.cat([sim_q_d_logit_scale, sim_q_q_logit_scale, sim_d_d_logit_scale]).max()

    sim_q_d = torch.exp(sim_q_d_logit_scale - max_val)
    sim_q_q = torch.exp(sim_q_q_logit_scale - max_val)
    sim_d_d = torch.exp(sim_d_d_logit_scale - max_val)

    sim_d_q = sim_q_d.T
    z1 = sim_q_d.sum(dim=1, keepdim=True)
    z2 = sim_d_q.sum(dim=1, keepdim=True)

    # z3 = sum(exp(q_i, q_j)) for i != j
    z3 = sim_q_q
    # zero out the diagonal -> will always be 1 across the diagonal since q_i == q_i
    z3 = z3 - torch.diag_embed(torch.diagonal(z3, dim1=-2, dim2=-1))
    z3 = z3.sum(dim=1, keepdim=True)

    z4 = sim_d_d
    z4 = z4 - torch.diag_embed(torch.diagonal(z4, dim1=-2, dim2=-1))
    z4 = z4.sum(dim=1, keepdim=True)

    z = z1 + z2 + z3 + z4

    softmax_q_d = sim_q_d / z

    # could also do .diag()
    loss = -torch.log(softmax_q_d[indices, indices]).mean()

    return loss


def get_chunked_embeddings(model, chunks, fp8_recipe=None):
    embeddings = []
    if fp8_recipe is not None:
        context = partial(te.fp8_autocast, enabled=True, fp8_recipe=fp8_recipe)
    else:
        context = nullcontext

    with torch.no_grad():
        for chunk in chunks:
            with context():
                emb = model(**chunk)
            embeddings.append(emb["embedding"])

    return torch.concat(embeddings, dim=0)


def accumulate_gradients(model, inputs, cache, fp8_recipe=None):
    length = len(inputs)
    sync_contexts = [model.no_sync for _ in range(length - 1)] + [nullcontext]

    if fp8_recipe is not None:
        precision_context = partial(te.fp8_autocast, enabled=True, fp8_recipe=fp8_recipe)
    else:
        precision_context = nullcontext

    for inp, grad, sync_context in zip(inputs, cache, sync_contexts):
        with sync_context():
            with precision_context():
                embedding = model(**inp)["embedding"]
            surrogate = torch.dot(embedding.flatten(), grad.flatten())
            surrogate.backward()


def cache_loss(model, query_embeddings, document_embeddings, logit_scale, loss_fn_name="clip"):
    # only require grad for embedding / representation
    query_embs = query_embeddings.detach().requires_grad_()
    document_embs = document_embeddings.detach().requires_grad_()

    if loss_fn_name == "clip":
        loss_fn = clip_loss
    elif loss_fn_name == "gte":
        loss_fn = gte_loss

    with model.no_sync():
        loss = loss_fn(query_embs, document_embs, logit_scale, gather_enabled=True)
        loss.backward()

    query_cache = query_embs.grad
    document_cache = document_embs.grad

    return query_cache, document_cache, loss.detach()


def grad_cache_loss_biencoder(model, inputs, chunk_size, logit_scale, use_fp8=False, loss_fn_name="clip"):
    chunked_query_inputs = []
    chunked_document_inputs = []

    query_bs = inputs["query_input_ids"].shape[0]
    for chunk_start in range(0, query_bs, chunk_size):
        chunk_end = min(query_bs, chunk_start + chunk_size)
        query_chunk = {
            k.replace("query_", ""): v[chunk_start:chunk_end] for k, v in inputs.items() if k.startswith("query_")
        }
        chunked_query_inputs.append(query_chunk)

    # we need to do this in case we have added hard negatives for the finetuning stage
    document_bs = inputs["document_input_ids"].shape[0]
    for chunk_start in range(0, document_bs, chunk_size):
        chunk_end = min(document_bs, chunk_start + chunk_size)
        document_chunk = {
            k.replace("document_", ""): v[chunk_start:chunk_end] for k, v in inputs.items() if k.startswith("document_")
        }
        chunked_document_inputs.append(document_chunk)

    if use_fp8:
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format,
            amax_history_len=16,
            amax_compute_algo="max",
        )
    else:
        fp8_recipe = None

    query_embs = get_chunked_embeddings(model, chunked_query_inputs, fp8_recipe=fp8_recipe)
    document_embs = get_chunked_embeddings(model, chunked_document_inputs, fp8_recipe=fp8_recipe)

    query_cache, document_cache, loss = cache_loss(
        model, query_embs, document_embs, logit_scale, loss_fn_name=loss_fn_name
    )

    chunked_query_cache = query_cache.split(chunk_size)
    chunked_document_cache = document_cache.split(chunk_size)

    accumulate_gradients(model, chunked_query_inputs, chunked_query_cache)
    accumulate_gradients(model, chunked_document_inputs, chunked_document_cache)

    return loss


def cache_loss_image_text(
    text_model, text_embs, vision_model, vision_embs, logit_scale, loss_fn_name="clip", bidirectional=True
):
    # only require grad for embedding / representation
    vision_embs = vision_embs.detach().requires_grad_()
    text_embs = text_embs.detach().requires_grad_()

    if loss_fn_name == "clip":
        loss_fn = clip_loss
    elif loss_fn_name == "gte":
        loss_fn = gte_loss

    no_text_sync = text_model.no_sync if hasattr(text_model, "no_sync") else nullcontext
    no_vision_sync = vision_model.no_sync if hasattr(vision_model, "no_sync") else nullcontext

    with no_text_sync():
        with no_vision_sync():
            loss = loss_fn(text_embs, vision_embs, logit_scale, gather_enabled=True, bidirectional=bidirectional)
            loss.backward()

    text_cache = text_embs.grad
    vision_cache = vision_embs.grad

    return text_cache, vision_cache, loss.detach()


def grad_cache_loss_image_text(
    text_model,
    vision_model,
    text_inputs,
    vision_inputs,
    chunk_size,
    logit_scale,
    use_fp8=False,
    loss_fn_name="clip",
    bidirectional=True,
):
    total_bs = text_inputs["input_ids"].shape[0]
    chunked_text = []
    chunked_vision = []

    for chunk_start in range(0, total_bs, chunk_size):
        text_chunk = {k: v[chunk_start : chunk_start + chunk_size] for k, v in text_inputs.items()}
        chunked_text.append(text_chunk)

        vision_chunk = {k: v[chunk_start : chunk_start + chunk_size] for k, v in vision_inputs.items()}
        chunked_vision.append(vision_chunk)

    if use_fp8:
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format,
            amax_history_len=16,
            amax_compute_algo="max",
        )
    else:
        fp8_recipe = None

    text_embs = get_chunked_embeddings(text_model, chunked_text, fp8_recipe=fp8_recipe)
    vision_embs = get_chunked_embeddings(vision_model, chunked_vision, fp8_recipe=fp8_recipe)

    text_cache, vision_cache, loss = cache_loss_image_text(
        text_model=text_model,
        vision_model=vision_model,
        vision_embs=vision_embs,
        text_embs=text_embs,
        logit_scale=logit_scale,
        loss_fn_name=loss_fn_name,
        bidirectional=bidirectional,
    )

    chunked_text_cache = text_cache.split(chunk_size)
    accumulate_gradients(text_model, chunked_text, chunked_text_cache)

    # in case second model is frozen, don't accumulate gradients
    if hasattr(vision_model, "no_sync"):
        chunked_vision_cache = vision_cache.split(chunk_size)
        accumulate_gradients(vision_model, chunked_vision, chunked_vision_cache)

    return loss
