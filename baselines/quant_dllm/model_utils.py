"""Shared model hooks used by the LLaDA and Dream quantizers."""

import functools

import torch


class FPInputsCache:
    """Capture full-precision inputs for each quantized projection group."""

    def __init__(self, sequential):
        self.names = [name for group in sequential for name in group]
        self.fp_cache = {name: [] for name in self.names}
        self.handles = []

    def cache_fp_input(self, module, inputs, output, name):
        del module, output
        value = inputs[0].detach()
        if value.ndim == 3:
            value = value.reshape((-1, value.shape[-1]))
        self.fp_cache[name].append(value.t())

    def add_hook(self, layers):
        for name in self.names:
            self.handles.append(
                layers[name].register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)
                )
            )

    def clear_hook(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        torch.cuda.empty_cache()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name].clear()
