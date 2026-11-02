"""
Monkey-patch accelerate's AlignDevicesHook to support expert pinning.

When an expert's hook has `pin = True`, the expert stays on GPU after the
first forward pass (offloading is disabled).
"""

from accelerate.hooks import AlignDevicesHook

_original_init = AlignDevicesHook.__init__
_original_pre_forward = AlignDevicesHook.pre_forward


def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.pin = False


def _patched_pre_forward(self, module, *args, **kwargs):
    result = _original_pre_forward(self, module, *args, **kwargs)
    if self.pin:
        self.offload = False
    return result


def patch():
    AlignDevicesHook.__init__ = _patched_init
    AlignDevicesHook.pre_forward = _patched_pre_forward
