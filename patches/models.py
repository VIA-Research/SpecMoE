"""
Register custom model classes (with fixed_gating support) with HuggingFace's
AutoModel, so from_pretrained() automatically uses our modified implementations.
"""

from transformers import AutoModelForSeq2SeqLM
from transformers.models.nllb_moe.configuration_nllb_moe import NllbMoeConfig


def patch():
    from modelings.nllb import NllbMoeForConditionalGeneration

    AutoModelForSeq2SeqLM.register(
        NllbMoeConfig, NllbMoeForConditionalGeneration, exist_ok=True
    )
