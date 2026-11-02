"""Generation settings shared by speculative decoding and HF generation."""

from dataclasses import dataclass
import math

import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


@dataclass(frozen=True)
class DecodingConfig:
    mode: str = "greedy"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self):
        if self.mode not in ("greedy", "sampling"):
            raise ValueError("decoding must be 'greedy' or 'sampling'")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("sampling temperature must be finite and greater than zero")
        if self.top_k < 0:
            raise ValueError("top-k must be non-negative (0 disables it)")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top-p must be in (0, 1]")
        if self.mode == "greedy" and (self.top_k != 0 or self.top_p != 1):
            raise ValueError("top-k and top-p require --decoding=sampling")

    @property
    def greedy(self):
        return self.mode == "greedy"

    @property
    def effective_temperature(self):
        return 0.0 if self.greedy else self.temperature

    def log_probs(self, logits):
        shape = logits.shape
        scores = logits.float().reshape(-1, shape[-1])
        processors = LogitsProcessorList()
        if not self.greedy:
            if self.temperature != 1:
                processors.append(TemperatureLogitsWarper(self.temperature))
            if self.top_k:
                processors.append(TopKLogitsWarper(self.top_k))
            if self.top_p < 1:
                processors.append(TopPLogitsWarper(self.top_p))
        scores = processors(None, scores)
        return torch.log_softmax(scores, dim=-1).reshape(shape)

    def generation_config(self, model_config, tokenizer, max_new_tokens,
                          forced_bos_token_id=None):
        # Construct a fresh config so checkpoint generation defaults cannot add
        # beam search, truncation, penalties, or a forced final EOS to one path.
        sampling = {} if self.greedy else {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        return GenerationConfig(
            do_sample=not self.greedy,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            bos_token_id=model_config.bos_token_id,
            decoder_start_token_id=model_config.decoder_start_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=None,
            **sampling,
        )
