import torch
from torch.optim import AdamW


# adapted from https://github.com/karpathy/minGPT/commit/bbbdac74fa9b2e55574d70056163ffbae42310c1#diff-2075fa9c224b395be5bda85544dd36572b59c76c54562819eadadbf268602834R157s
# and using similar logic from openclip
def configure_optimizer(modules, args):
    decay = set()
    no_decay = set()
    blacklist_weight_modules = (torch.nn.LayerNorm,)
    named_parameters = [(name, param) for model in modules for name, param in model.named_parameters()]
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        # YUCK!!!
        if param.squeeze().ndim < 2:
            no_decay.add(name)
        elif "bias" in name:
            no_decay.add(name)
        elif isinstance(param, blacklist_weight_modules):
            no_decay.add(name)
        elif "logit_scale" in name:
            no_decay.add(name)
        else:
            decay.add(name)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in named_parameters if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "lr": args.learning_rate},
        # if we want different learning rates for the projection layer and encoder
    ]

    optimizer = AdamW(optim_groups, betas=(args.adam_beta1, args.adam_beta2), eps=args.eps)
    return optimizer
