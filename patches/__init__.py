"""
Runtime registration and hooks for SpecMoE inference.

Call apply_patches() before any model loading to activate:
  - Expert pinning support in accelerate hooks
  - Custom NLLB-MoE model class with fixed_gating
  - Optional measurement instrumentation (overlap/movement profiling)
"""

from . import accelerate_hooks
from . import models


def apply_patches(measurement_mode=None):
    accelerate_hooks.patch()
    models.patch()

    if measurement_mode:
        from . import measurement
        measurement.patch(measurement_mode)
