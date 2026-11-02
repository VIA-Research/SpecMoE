# SPDX-License-Identifier: MIT
"""SpecMoE extensions composed with the installed Transformers NLLB model."""

from contextvars import ContextVar
from functools import wraps
import inspect
import json
from pathlib import Path

import torch
from transformers.models.nllb_moe import modeling_nllb_moe as upstream


ROOT = Path(__file__).resolve().parents[1]
AFFINITY_DIR = ROOT / "affinity_tables"
HOT_EXPERTS = ROOT / "hot_experts" / "nllb_54b.json"


def _router_inputs(module, args, kwargs):
    """Give the upstream router one padding flag per current token."""
    states = args[0] if args else kwargs["hidden_states"]
    mask = args[1] if len(args) > 1 else kwargs.get("padding_mask")
    if isinstance(mask, torch.Tensor) and mask.ndim == 4:
        mask = mask[:, 0, -1, -states.shape[1]:].reshape(-1)
        if len(args) > 1:
            args = (args[0], mask, *args[2:])
        else:
            kwargs = {**kwargs, "padding_mask": mask}
    return args, kwargs


class ExpertRouting:
    """Select and execute active experts without copying the upstream MLP."""

    def __init__(self, module, draft_mode, layer_idx=None):
        self.module = module
        self.draft_mode = draft_mode
        self.layer_idx = layer_idx
        self.num_fixed_experts = 4
        self.fixed_expert = None
        self.affinity_table = None
        self.affinity_expert_map = None
        module.router.register_forward_pre_hook(_router_inputs, with_kwargs=True)
        module.forward = self.forward

    def _pin(self, indices, enabled):
        for index in indices.tolist():
            for child in self.module.experts[f"expert_{index}"].children():
                hook = getattr(child, "_hf_hook", None)
                if hook is not None:
                    hook.pin = enabled
                    if not enabled:
                        hook.offload = True

    def init_fixed_expert(self, count):
        size = len(self.module.experts)
        if not 1 <= count <= size:
            raise ValueError(f"Draft expert count must be in [1, {size}].")
        ranking = json.loads(HOT_EXPERTS.read_text())["decoder"][self.layer_idx // 4]
        ranking = [index for index in ranking if index < size]
        ranking += [index for index in range(size) if index not in ranking]
        self.fixed_expert = torch.tensor(ranking[:count])
        self._pin(self.fixed_expert, True)

    @staticmethod
    def _weight(linear):
        hook = getattr(linear, "_hf_hook", None)
        if hook is not None and hook.weights_map is not None:
            return hook.weights_map["weight"]
        return linear.weight.detach().cpu()

    def get_affinity_table(self):
        if self.affinity_table is None:
            path = AFFINITY_DIR / f"affinity_nllb_{self.layer_idx}.pt"
            size = len(self.module.experts)
            if path.is_file():
                self.affinity_table = torch.load(path, map_location="cpu", weights_only=True)
                if self.affinity_table.shape != (size, size):
                    raise ValueError(f"Affinity shape does not match {size} experts: {path}")
            else:
                self.affinity_table = torch.zeros(size, size)
                for left in range(size):
                    for right in range(left + 1, size):
                        first = self.module.experts[f"expert_{left}"]
                        second = self.module.experts[f"expert_{right}"]
                        distance = sum(torch.linalg.vector_norm(
                            self._weight(getattr(first, name)) - self._weight(getattr(second, name))
                        ) for name in ("fc1", "fc2"))
                        self.affinity_table[left, right] = distance
                        self.affinity_table[right, left] = distance
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.affinity_table, path)
        return self.affinity_table

    def update_affinity_expert_map(self):
        ordered = self.get_affinity_table().argsort(dim=-1)
        pinned = set(self.fixed_expert.tolist())
        self.affinity_expert_map = [
            [index for index in row if index in pinned] for row in ordered.tolist()
        ]

    def _draft_weights(self, states, padding_mask):
        if self.fixed_expert is None:
            self.init_fixed_expert(self.num_fixed_experts)
            if self.num_fixed_experts > 2:
                self.update_affinity_expert_map()
        rows = states.shape[0] * states.shape[1]
        weights = states.new_zeros(rows, len(self.module.experts))
        if self.num_fixed_experts <= 2:
            weights[:, self.fixed_expert.to(states.device)] = 0.5
            return weights

        _, probabilities = self.module.router(states, padding_mask)
        selected = [[] for _ in range(rows)]
        for token, expert in probabilities.nonzero().tolist():
            selected[token].append(expert)
        tokens, experts = [], []
        for token, original in enumerate(selected):
            for position, expert in enumerate(original):
                peer = original[1 - position] if len(original) == 2 else None
                replacement = next(index for index in self.affinity_expert_map[expert] if index != peer)
                tokens.append(token)
                experts.append(replacement)
        # Preserve the published draft policy: each mapped expert has weight
        # 1/2, including when two original routes map to the same expert.
        weights[tokens, experts] = 0.5
        return weights

    def _replace_pinned(self, probabilities):
        if self.fixed_expert is None:
            return
        replacement = probabilities.ne(0).sum(0).topk(self.num_fixed_experts).indices.cpu()
        self._pin(self.fixed_expert, False)
        self._pin(replacement, True)
        self.fixed_expert = replacement
        if self.num_fixed_experts > 2:
            self.update_affinity_expert_map()

    def forward(self, hidden_states, padding_mask=None):
        draft = self.layer_idx is not None and self.draft_mode.get()
        if draft:
            coefficients = self._draft_weights(hidden_states, padding_mask)
            router_result = (None, None)
        else:
            first_choice, coefficients = self.module.router(hidden_states, padding_mask)
            router_result = (coefficients, first_choice.argmax(-1))
            if self.layer_idx is not None:
                self._replace_pinned(coefficients)

        # A token has at most two routes. Accumulate in two slots so fp16
        # addition rounds once, and never call an expert with an empty batch.
        flat = hidden_states.flatten(0, 1)
        active = coefficients.ne(0)
        slots = active.long().cumsum(-1) - 1
        contributions = flat.new_zeros(2, *flat.shape)
        for expert_id in active.any(0).nonzero().flatten().tolist():
            tokens = active[:, expert_id].nonzero().flatten()
            values = self.module.experts[f"expert_{expert_id}"](flat.index_select(0, tokens))
            if self.module.training:
                values = self.module.token_dropout(values)
            elif self.module.moe_token_dropout:
                values = values * (1 - self.module.moe_token_dropout)
            contributions[slots[tokens, expert_id], tokens] = values * coefficients[tokens, expert_id, None]
        return contributions.sum(0).reshape_as(hidden_states), router_result


def extend_decoder(decoder, draft_mode=None):
    """Attach per-instance routing and cache-position extensions."""
    if hasattr(decoder, "_specmoe_routing"):
        return decoder._specmoe_routing
    if draft_mode is None:
        draft_mode = ContextVar("nllb_draft_mode", default=False)
    current_mask = ContextVar("nllb_decoder_mask", default=None)
    decoder_forward = decoder.forward
    decoder_signature = inspect.signature(decoder_forward)

    @wraps(decoder_forward)
    def decode(*args, **kwargs):
        values = decoder_signature.bind(*args, **kwargs).arguments
        token = current_mask.set(values.get("attention_mask"))
        try:
            return decoder_forward(*args, **kwargs)
        finally:
            current_mask.reset(token)

    embedding = decoder.embed_positions
    embed_forward = embedding.forward
    embed_signature = inspect.signature(embed_forward)

    @wraps(embed_forward)
    def positions(*args, **kwargs):
        result = embed_forward(*args, **kwargs)
        mask = current_mask.get()
        if mask is None or mask.ndim != 2:
            return result
        values = embed_signature.bind(*args, **kwargs).arguments
        past = values.get("past_key_values_length", 0)
        offsets = mask[:, :past].long().sum(-1, keepdim=True)
        ids = values.get("input_ids")
        if ids is not None:
            indices = upstream.create_position_ids_from_input_ids(ids, embedding.padding_idx, offsets)
        else:
            indices = embedding.create_position_ids_from_inputs_embeds(values["inputs_embeds"], offsets)
        return torch.nn.functional.embedding(indices.to(embedding.weights.device), embedding.weights).detach()

    decoder.forward = decode
    embedding.forward = positions
    decoder._specmoe_routing = [
        ExpertRouting(layer.ffn, draft_mode, index)
        for index, layer in enumerate(decoder.layers) if layer.is_sparse
    ]
    return decoder._specmoe_routing


class NllbMoeForConditionalGeneration(upstream.NllbMoeForConditionalGeneration):
    """The installed NLLB model plus SpecMoE's inference extensions."""

    def __init__(self, config):
        super().__init__(config)
        self._draft_mode = ContextVar("nllb_draft_mode", default=False)
        self._decoder_routing = extend_decoder(self.model.decoder, self._draft_mode)
        self._encoder_routing = [
            ExpertRouting(layer.ffn, self._draft_mode)
            for layer in self.model.encoder.layers if layer.is_sparse
        ]

    def set_num_fixed_experts(self, count):
        if not 1 <= count <= self.config.num_experts:
            raise ValueError(f"Draft expert count must be in [1, {self.config.num_experts}].")
        for routing in self._decoder_routing:
            if routing.fixed_expert is not None:
                routing._pin(routing.fixed_expert, False)
            routing.num_fixed_experts = count
            routing.fixed_expert = None
            routing.affinity_expert_map = None

    @wraps(upstream.NllbMoeForConditionalGeneration.forward)
    def forward(self, *args, fixed_gating=False, logits_to_keep=None, **kwargs):
        token = self._draft_mode.set(fixed_gating)
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._draft_mode.reset(token)
