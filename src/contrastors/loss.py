from contextlib import nullcontext
from functools import partial

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from contrastors.distributed import gather, gather_dict


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
    tracker=None,
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

            tracker.log({"kd_loss": kl_loss.detach().cpu().item(), "ce_loss": loss.detach().cpu().item()}, step=step)

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
            tracker.log({"cross_encoder_tabe": table}, step=step)

            loss = loss * alpha + kl_loss

    else:
        similarity_query_document = logit_scale(torch.matmul(query, document.T))
        labels = labels * (document.size(0) // query.size(0))
        if bidirectional:
            similarity_document_query = logit_scale(torch.matmul(document, query.T))
            loss = (
                F.cross_entropy(similarity_query_document, labels) + F.cross_entropy(similarity_document_query, labels)
            ) * dist.get_world_size()
        else:
            loss = F.cross_entropy(similarity_query_document, labels) * dist.get_world_size()

    if tracker is not None:
        accuracy = (similarity_query_document.argmax(dim=1) == labels).float().mean()
        tracker.log({f"accuracy_{dataset}": accuracy.detach().cpu().item()}, step=step)

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


def get_chunked_embeddings(model, chunks):
    embeddings = []

    with torch.no_grad():
        for chunk in chunks:
            emb = model(**chunk)
            embeddings.append(emb["embedding"])

    return torch.concat(embeddings, dim=0)


def accumulate_gradients(model, inputs, cache):
    length = len(inputs)
    sync_contexts = [model.no_sync for _ in range(length - 1)] + [nullcontext]

    for inp, grad, sync_context in zip(inputs, cache, sync_contexts):
        with sync_context():
            embedding = model(**inp)["embedding"]
            surrogate = torch.dot(embedding.flatten(), grad.flatten())
            surrogate.backward()


def cache_loss(tower1, tower2, query_embeddings, document_embeddings, logit_scale, bidirectional=False):
    # only require grad for embedding / representation
    query_embs = query_embeddings.detach().requires_grad_()
    document_embs = document_embeddings.detach().requires_grad_()

    no_tower1_sync = tower1.no_sync if tower1.training else nullcontext
    no_tower2_sync = tower2.no_sync if tower2.training else nullcontext

    with no_tower1_sync():
        with no_tower2_sync():
            loss = clip_loss(query_embs, document_embs, logit_scale, gather_enabled=True, bidirectional=bidirectional)
            loss.backward()

    query_cache = query_embs.grad
    document_cache = document_embs.grad

    return query_cache, document_cache, loss.detach()


def grad_cache_loss(tower1, t1_inputs, tower2, t2_inputs, chunk_size, logit_scale, bidirectional=False):
    total_bs = t1_inputs["input_ids"].shape[0]
    chunked_queries = []
    chunked_documents = []

    for chunk_start in range(0, total_bs, chunk_size):
        query_chunk = {k: v[chunk_start : chunk_start + chunk_size] for k, v in t1_inputs.items()}
        chunked_queries.append(query_chunk)

        document_chunk = {k: v[chunk_start : chunk_start + chunk_size] for k, v in t2_inputs.items()}
        chunked_documents.append(document_chunk)

    query_embs = get_chunked_embeddings(tower1, chunked_queries)
    document_embs = get_chunked_embeddings(tower2, chunked_documents)

    query_cache, document_cache, loss = cache_loss(
        tower1, tower2, query_embs, document_embs, logit_scale, bidirectional=bidirectional
    )

    chunked_query_cache = query_cache.split(chunk_size)
    chunked_document_cache = document_cache.split(chunk_size)

    accumulate_gradients(tower1, chunked_queries, chunked_query_cache)
    if tower2.training:
        accumulate_gradients(tower2, chunked_documents, chunked_document_cache)

    return loss
