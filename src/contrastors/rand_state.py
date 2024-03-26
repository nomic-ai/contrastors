import torch
from torch.utils.checkpoint import get_device_states, set_device_states


# taken from: https://github.com/luyug/GradCache/blob/0c33638cb27c2519ad09c476824d550589a8ec38/src/grad_cache/context_managers.py
class RandContext:
    def __init__(self, tensors):
        if isinstance(tensors, dict):
            tensors = list(tensors.values())
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None
