"""
Measurement patches for profiling expert offloading.

Wraps accelerate's set_module_tensor_to_device to silently accumulate
per-transfer bytes (movement mode) or wall-clock time (overlap mode).
Call reset() before a batch and report() after to read the totals.
"""

import time
import torch
import accelerate.hooks as _acc_hooks
import accelerate.utils.modeling as _acc_modeling

# accelerate.hooks and accelerate.utils each cache their own reference to
# set_module_tensor_to_device via `from ... import ...` at import time, so we
# must rebind the name in every module that already pulled it in.
_patch_targets = [_acc_modeling, _acc_hooks]
_original_set_module_tensor = _acc_modeling.set_module_tensor_to_device

_stats = {"bytes": 0, "ms": 0.0, "count": 0}


def reset():
    _stats["bytes"] = 0
    _stats["ms"] = 0.0
    _stats["count"] = 0


def report():
    return dict(_stats)


def _overlap_set_module_tensor(module, tensor_name, device, value=None, **kwargs):
    if value is not None and isinstance(value, torch.Tensor) and str(device) != "meta":
        tik = time.perf_counter()
        result = _original_set_module_tensor(module, tensor_name, device, value=value, **kwargs)
        torch.cuda.synchronize()
        tok = time.perf_counter()
        _stats["ms"] += (tok - tik) * 1000
        _stats["count"] += 1
        return result
    return _original_set_module_tensor(module, tensor_name, device, value=value, **kwargs)


def _movement_set_module_tensor(module, tensor_name, device, value=None, **kwargs):
    if value is not None and isinstance(value, torch.Tensor) and str(device) != "meta":
        _stats["bytes"] += value.numel() * value.element_size()
        _stats["count"] += 1
    return _original_set_module_tensor(module, tensor_name, device, value=value, **kwargs)


def patch(mode):
    if mode == "overlap":
        wrapper = _overlap_set_module_tensor
    elif mode == "movement":
        wrapper = _movement_set_module_tensor
    else:
        raise ValueError(f"Unknown measurement mode: {mode}. Use 'overlap' or 'movement'.")
    for target in _patch_targets:
        target.set_module_tensor_to_device = wrapper
