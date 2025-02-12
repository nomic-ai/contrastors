import torch.distributed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, XLMRobertaModel
from contrastors import BiEncoder, BiEncoderConfig


from sentence_transformers import SentenceTransformer
from .text_text import TextTextTrainer, SentenceTransformerModule
from contrastors.distributed import gather_with_grad

from torch import nn
import re


# adapted from https://github.com/OscarXZQ/weight-selection/blob/e9b6d0ca0cd288d551c6aa25645863804fc63633/weight_selection.py
def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim]-1, s_shape[dim])).long()
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == s_shape
    return ws

    
def transfer_weights(teacher, student, uniform_element_selection):
    """
    Transfer weights from teacher to student model, handling dimension reduction
    and layer reduction.
    
    Args:
        teacher: Teacher model with original architecture
        student: Student model with reduced dimensions/layers
        uniform_element_selection: Function that handles dimension reduction
    """
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    new_state_dict = {}

    # Get layer mapping (e.g., if teacher has 12 layers and student has 6,
    # we'll take layers 0,2,4,6,8,10 from teacher)
    teacher_layers = teacher.trunk.config.num_hidden_layers
    student_layers = student.trunk.config.num_hidden_layers
    layer_mapping = {i: i * 2 for i in range(student_layers)}

    for key in student_state.keys():
        if key not in teacher_state:
            continue
            
        # Handle layer mapping for transformer blocks
        layer_match = re.search(r'trunk\.encoder.layers\.(\d+)\.', key)
        if layer_match:
            student_layer_idx = int(layer_match.group(1))
            if student_layer_idx >= student_layers:
                continue
                
            # Create the corresponding teacher key
            teacher_layer_idx = layer_mapping[student_layer_idx]
            teacher_key = key.replace(
                f'trunk.encoder.layers.{student_layer_idx}.',
                f'trunk.encoder.layers.{teacher_layer_idx}.'
            )
        else:
            teacher_key = key

        # Get the weights
        breakpoint()
        teacher_weights = teacher_state[teacher_key]
        student_shape = student_state[key].shape

        # If shapes match, copy directly
        if teacher_weights.shape == student_shape:
            new_state_dict[key] = teacher_weights
        # If shapes don't match, use uniform_element_selection
        else:
            new_state_dict[key] = uniform_element_selection(
                teacher_weights, 
                student_shape
            )

    # Load the new state dict into student model
    breakpoint()
    student.load_state_dict(new_state_dict, strict=False)
    return student

    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class DistillTrainer(TextTextTrainer):
    def __init__(self, config, dtype):
        super(TextTextTrainer, self).__init__(config, dtype)

        self.loss_fn = config.train_args.distill_loss_fn

    def get_model(self, config):
        teacher_name = config.model_args.model_name
        if config.model_args.checkpoint is not None:
            model_config = BiEncoderConfig.from_pretrained(config.model_args.checkpoint)
            teacher = BiEncoder.from_pretrained(config.model_args.checkpoint, config=model_config)
            teacher_config = teacher.trunk.config
            # biencoder is a wrapper around encoder -> pooling + normalization
            teacher_weights = teacher.trunk

            student_config = AutoConfig.from_pretrained("nomic-ai/nomic-xlm-2048", trust_remote_code=True)
            student_config.num_hidden_layers = teacher_config.num_hidden_layers // 2
            # student_config.hidden_size = teacher_config.hidden_size // config.model_args.ffn_div
            # student_config.n_inner = getattr(teacher_config, "n_inner", None) // config.model_args.ffn_div

            student = AutoModel.from_config(student_config, trust_remote_code=True, add_pooling_layer=False)
            teacher_sd = teacher_weights.state_dict()
            student_sd = student.state_dict()
            for key in student.state_dict():
                layer_match = re.search(r'encoder.layers\.(\d+)\.', key)

                if layer_match:
                    student_layer_idx = int(layer_match.group(1))

                    # Create the corresponding teacher key
                    teacher_layer_idx = student_layer_idx * 2
                    teacher_key = key.replace(
                        f'encoder.layers.{student_layer_idx}.',
                        f'encoder.layers.{teacher_layer_idx}.'
                    )
                else:
                    teacher_key = key
                
                teacher_weight = teacher_sd[teacher_key]
                student_sd[key] = uniform_element_selection(teacher_weight, student_sd[key].shape)

            student.load_state_dict(student_sd)

            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False

            if config.train_args.distill_loss_fn in ["", "", "towers"]:
                student_projection = nn.Linear(student_config.hidden_size, teacher_config.hidden_size)
                # student_projection = MLP(student_config.hidden_size, 2*teacher_config.hidden_size, teacher_config.hidden_size)

        else:
            teacher = AutoModel.from_pretrained(teacher_name, add_pooling_layer=False)
            teacher_config = teacher.config
            teacher_weights = teacher
            # freeze teacher
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            student_config = AutoConfig.from_pretrained(teacher_name)
            # student_config.num_hidden_layers = teacher_config.num_hidden_layers // 2
            # set dropout to 0
            student_config.hidden_dropout_prob = 0
            student_config.attention_probs_dropout_prob = 0
            if config.model_args.ffn_div is not None:
                student_config.hidden_size = teacher_config.hidden_size // config.model_args.ffn_div
                student_config.intermediate_size = getattr(teacher_config, "intermediate_size", None) or getattr(teacher_config, "n_inner", None) // config.model_args.ffn_div
                # student_config.num_attention_heads = teacher.config.num_attention_heads // config.model_args.ffn_div

                # projection back to teacher size
                if config.train_args.distill_loss_fn in ["", "", "towers"]:
                    # student_projection = nn.Linear(student_config.hidden_size, teacher.config.hidden_size)
                    student_projection = MLP(student_config.hidden_size, 2*teacher_config.hidden_size, teacher_config.hidden_size)

            student = XLMRobertaModel(student_config, add_pooling_layer=False)

            if config.model_args.gradient_checkpointing:
                student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if self.config.model_args.distill_init_pretrained:
                if config.model_args.ffn_div is not None:
                    with torch.no_grad():
                        # Word embeddings
                        student.embeddings.word_embeddings.weight.data = uniform_element_selection(
                            teacher.embeddings.word_embeddings.weight.data,
                            student.embeddings.word_embeddings.weight.shape
                        )
                        
                        # Position embeddings
                        if hasattr(teacher.embeddings, "position_embeddings"):
                            student.embeddings.position_embeddings.weight.data = uniform_element_selection(
                                teacher.embeddings.position_embeddings.weight.data,
                                student.embeddings.position_embeddings.weight.shape
                            )
                        
                        # Token type embeddings
                        student.embeddings.token_type_embeddings.weight.data = uniform_element_selection(
                            teacher.embeddings.token_type_embeddings.weight.data,
                            student.embeddings.token_type_embeddings.weight.shape
                        )

                for i in range(student_config.num_hidden_layers):
                    teacher_layer = teacher.encoder.layer[i//2]
                    student_layer = student.encoder.layer[i]
                    
                    with torch.no_grad():
                        # If ffn_div is specified, apply uniform selection to weights
                        if config.model_args.ffn_div is not None:
                            # Attention weights
                            student_layer.attention.self.query.weight.data = uniform_element_selection(
                                teacher_layer.attention.self.query.weight.data,
                                student_layer.attention.self.query.weight.shape
                            )
                            student_layer.attention.self.key.weight.data = uniform_element_selection(
                                teacher_layer.attention.self.key.weight.data,
                                student_layer.attention.self.key.weight.shape
                            )
                            student_layer.attention.self.value.weight.data = uniform_element_selection(
                                teacher_layer.attention.self.value.weight.data,
                                student_layer.attention.self.value.weight.shape
                            )
                            student_layer.attention.output.dense.weight.data = uniform_element_selection(
                                teacher_layer.attention.output.dense.weight.data,
                                student_layer.attention.output.dense.weight.shape
                            )
                            
                            # FFN weights
                            student_layer.intermediate.dense.weight.data = uniform_element_selection(
                                teacher_layer.intermediate.dense.weight.data,
                                student_layer.intermediate.dense.weight.shape
                            )
                            student_layer.output.dense.weight.data = uniform_element_selection(
                                teacher_layer.output.dense.weight.data,
                                student_layer.output.dense.weight.shape
                            )
                        else:
                            # If no dimension reduction, just copy the weights
                            student_layer.load_state_dict(teacher_layer.state_dict(), strict=False)

        if self.distributed and not self.deepspeed:
            student = student.to("cuda")
            student = torch.nn.parallel.DistributedDataParallel(
                student,
                device_ids=[self.process_index],
                # find_unused_parameters=True,
                broadcast_buffers=False,
            )
            if config.model_args.ffn_div is not None and config.train_args.distill_loss_fn in ["", "", "towers"]:
                student_projection = student_projection.to("cuda")
                student_projection = torch.nn.parallel.DistributedDataParallel(
                    student_projection,
                    device_ids=[self.process_index],
                    # find_unused_parameters=True,
                    broadcast_buffers=False,
                )

        models = {"model": student, "teacher": teacher.to("cuda")}
        if config.model_args.ffn_div is not None and config.train_args.distill_loss_fn in ["", "", "towers"]:
            models["projection"] = student_projection

        return models

    def save_model(self, output_dir):
        super().save_model(output_dir)
        if self.global_rank == 0:
            logit_scale = self.model.get("logit_scale", None)
            if isinstance(logit_scale, (nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel)) and any(
                p.requires_grad for p in logit_scale.parameters()
            ):
                unwrapped_scale = self.unwrap(logit_scale)
                torch.save(unwrapped_scale.state_dict(), f"{output_dir}/logit_scale.pt")

    def clip_gradients(self, max_grad_norm):
        super().clip_gradients(max_grad_norm)

    def infonce(self, query, document, temperature):
        document = gather_with_grad(document)
        device = query.device
        labels = torch.arange(query.shape[0]).to(device)
        similarity_query_document = torch.matmul(query, document.T) / temperature
        num_logits = similarity_query_document.size(0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        # calculate sub-batch labels
        labels = labels + rank * num_logits

        # if training with negatives
        # multiply by world size since we only gather the document embeddings
        labels = labels * (document.size(0) // (query.size(0) * dist.get_world_size()))

        loss = F.cross_entropy(similarity_query_document, labels) * dist.get_world_size()

        return loss

    def get_score_diff(self, vectors):
        scores = torch.matmul(vectors, vectors.T)
        scores = scores[torch.triu(torch.ones_like(scores), diagonal=1).bool()]
        score_diff = scores.reshape((1, -1)) - scores.reshape((-1, 1))
        score_diff = score_diff[torch.triu(torch.ones_like(score_diff), diagonal=1).bool()]
        return score_diff

    def forward_step(self, model, inputs, teacher, projection=None, **kwargs):
        model.train()

        query_inputs = {
                "input_ids": inputs["query_input_ids"].to(model.device), 
                "attention_mask": inputs["query_attention_mask"].to(model.device)
        }

        document_inputs = {
                "input_ids": inputs["document_input_ids"].to(model.device), 
                "attention_mask": inputs["document_attention_mask"].to(model.device)
        }

        with torch.no_grad():
            teacher_query_outputs = teacher(**query_inputs)["embedding"]
            teacher_document_outputs = teacher(**document_inputs)["embedding"]

            
        student_query_outputs = model(**query_inputs)[0]
        student_document_outputs = model(**document_inputs)[0]
        
        student_query_outputs = student_query_outputs[:, 0]
        student_document_outputs = student_document_outputs[:, 0]

        if projection is not None:
            student_query_outputs = projection(student_query_outputs)
            student_document_outputs = projection(student_document_outputs)

        norm_teacher_query = F.normalize(teacher_query_outputs, dim=-1)
        norm_teacher_document = F.normalize(teacher_document_outputs, dim=-1)
        norm_student_query = F.normalize(student_query_outputs, dim=-1)
        norm_student_document = F.normalize(student_document_outputs, dim=-1)
        if self.loss_fn == "mse":
            query_mse = F.mse_loss(norm_student_query, norm_teacher_query)
            document_mse = F.mse_loss(norm_student_document, norm_teacher_document)

            loss = {"loss": query_mse + document_mse, "query_mse": query_mse, "document_mse": document_mse}

        elif self.loss_fn == "kd":
            teacher_sim_qd = torch.matmul(norm_teacher_query, norm_teacher_document.T)  / self.config.train_args.distill_temperature

            student_sim_qd = torch.matmul(norm_student_query, norm_student_document.T) / self.config.train_args.distill_temperature

            student_log_softmax = F.log_softmax(student_sim_qd, dim=-1)
            teacher_probs = F.softmax(teacher_sim_qd, dim=-1)

            kd_loss = F.kl_div(student_log_softmax, teacher_probs, reduction="batchmean")

            # hardcode temp to 50 for now
            infonce_loss = self.infonce(norm_student_query, norm_student_document, 0.02)
            total_loss = dist.get_world_size() * 1000 * kd_loss + infonce_loss

            loss = {"loss": total_loss, "kd_loss": kd_loss, "infonce_loss": infonce_loss}

        elif self.loss_fn == "towers":
            # q_s -> d_s 
            temperature = self.config.train_args.distill_temperature
            student_contrast_loss = self.infonce(norm_student_query, norm_student_document, temperature) 

            # q_s -> q_t
            student_teacher_query_loss = self.infonce(norm_student_query, norm_teacher_query, temperature)
            # d_s -> d_t
            student_teacher_document_loss = self.infonce(norm_student_document, norm_teacher_document, temperature)
            # q_s -> d_t
            student_teacher_contrast_loss = self.infonce(norm_student_query, norm_teacher_document, temperature)

            total_loss = (
                student_contrast_loss + student_teacher_query_loss  + student_teacher_document_loss + student_teacher_contrast_loss
                ) / 4 

            loss = {
                "loss": total_loss, 
                "loss_infonce_student": student_contrast_loss,
                "loss_teacher_query": student_teacher_query_loss,
                "loss_teacher_document": student_teacher_document_loss,
                "loss_infonce_teacher": student_teacher_contrast_loss 
                }

        elif self.loss_fn == "stella":
            temperature = self.config.train_args.distill_temperature
            cos_loss_query = (1 - (norm_student_query * norm_teacher_query).sum(axis=1).mean()) * 10
            cos_loss_document = (1 - (norm_student_document * norm_teacher_document).sum(axis=1).mean()) * 10

            cos_loss = cos_loss_query + cos_loss_document

            sim_query = F.mse_loss(
                torch.matmul(norm_student_query, norm_student_query.T),
                torch.matmul(norm_teacher_query, norm_teacher_query.T),
            ) * 200

            sim_document = F.mse_loss(
                torch.matmul(norm_student_document, norm_student_document.T),
                torch.matmul(norm_teacher_document, norm_teacher_document.T),
            ) * 200

            sim_loss = sim_query + sim_document

            triplet_query_label = torch.where(self.get_score_diff(norm_teacher_query) < 0, 1, -1)
            triplet_query_loss = F.relu(self.get_score_diff(norm_student_query) * triplet_query_label + 0.015).mean() * 20

            triplet_document_label = torch.where(self.get_score_diff(norm_teacher_document) < 0, 1, -1)
            triplet_document_loss = F.relu(self.get_score_diff(norm_student_document) * triplet_document_label + 0.015).mean() * 20

            triplet_loss = triplet_query_loss + triplet_document_loss
            

            loss = {
                "loss": cos_loss + sim_loss + triplet_loss,
                "cos_loss_query": cos_loss_query,
                "cos_loss_document": cos_loss_document,
                "sim_loss_query": sim_query,
                "sim_loss_document": sim_document,
                "triplet_loss_query": triplet_query_loss,
                "triplet_loss_document": triplet_document_loss,
                "triplet_loss": triplet_loss
            }

        else:
            raise NotImplementedError(f"Loss function {self.loss_fn} not implemented")

        return loss

    def backward(self, loss):
        if isinstance(loss, dict):
            loss = loss["loss"]

        if self.deepspeed:
            self.engine.backward(loss)
            self.engine.step()
        else:
            # grad cache backprops in the loss function, becomes a noop
            loss.backward()

    def training_step(
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
        loss = super().training_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            train_args=train_args,
            total_num_steps=total_num_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        if train_args.clamp_logits:
            with torch.no_grad():
                self.model["scale"].module.logit_scale.clamp_(0, np.log(train_args.logit_max))

        if train_args.wandb:
            if isinstance(loss, dict):
                self.log({k: v.detach().cpu().item() for k, v in loss.items()}, step=step)

        return loss

    def eval_loop(self, model, dataloader, step, **kwargs):
        model.eval()
        train_args = self.config.train_args
        model_args = self.config.model_args
        if self.process_index == 0:
            original_model = model.module
            module = nn.Sequential(SentenceTransformerModule(model=original_model, max_seq_length=model_args.seq_len, tokenizer=self.tokenizer, pooling="cls"))
            emb = SentenceTransformer(modules=module, similarity_fn_name="cosine")
            results = dataloader(emb) 

            ndcg = {f'beir/{k.replace("Nano", "").replace("_cosine", "").lower()}': v for k, v in results.items() if "ndcg@10" in k}

            if train_args.wandb:
                self.log(ndcg, step=step) 
            else:
                self.print(ndcg)
            
            
        torch.distributed.barrier()