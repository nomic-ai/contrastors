import torch


def quick_gelu(input):
    return input * torch.sigmoid(1.702 * input)
