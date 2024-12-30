import torch
import torch.distributed.nn
import torch.distributed as dist

def gather_with_grad(t):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t

    if t.ndim == 0:
        t = t.unsqueeze(0)

    return torch.cat(torch.distributed.nn.all_gather(t), dim=0)


def gather(t):
    # torch.distributed.nn.all_gather scales by world size since the reduce op is SUM
    # https://github.com/pytorch/pytorch/issues/58005
    # only should use torch.distributed.nn.all_gather if we implement a `local_loss`
    # like: https://github.com/mlfoundations/open_clip/issues/616
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t

    if t.ndim == 0:
        t = t.unsqueeze(0)

    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    gathered[dist.get_rank()] = t
    return torch.cat(gathered, dim=0)


def gather_dict(d):
    # gathers a dict of tensors
    gathered = {}
    for k, v in d.items():
        gathered[k] = gather(v)
    return gathered


def all_gather_object(obj):
    # gather a pickle-able python object across all ranks
    objects = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(objects, obj)

    flattened_objects = []
    for obj in objects:
        if isinstance(obj, list):
            flattened_objects.extend(obj)
        else:
            flattened_objects.append(obj)

    return flattened_objects


def print_rank_zero(msg):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(msg)


def print_in_order(msg):
    if dist.is_initialized():
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                print(msg)
            dist.barrier()
    else:
        print(msg)


class DistributedWandbTracker:
    def __init__(self, tracker):
        self.tracker = tracker

    def log(self, metrics, **kwargs):
        if dist.get_rank() == 0:
            self.tracker.log(metrics, **kwargs)
