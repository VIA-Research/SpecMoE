import json
import os

_HOT_EXPERTS_DIR = os.path.join(os.path.dirname(__file__), "hot_experts")


def _load_hot_experts(filename, key):
    with open(os.path.join(_HOT_EXPERTS_DIR, filename)) as f:
        return json.load(f)[key]


def get_device_map_nllb_54b(offload_device, cuda=0, cache=False, num_cached_experts=13):
    device_map_nllb_54b = dict()
    moe_step = 4
    num_layers = 24

    device_map_nllb_54b["model.shared"] = cuda
    device_map_nllb_54b["lm_head"] = cuda
    device_map_nllb_54b["model.encoder.embed_tokens"] = cuda
    device_map_nllb_54b["model.encoder.embed_positions"] = cuda
    device_map_nllb_54b["model.encoder.layer_norm"] = cuda
    device_map_nllb_54b["model.decoder.embed_tokens"] = cuda
    device_map_nllb_54b["model.decoder.embed_positions"] = cuda
    device_map_nllb_54b["model.decoder.layer_norm"] = cuda

    hot_experts = _load_hot_experts("nllb_54b.json", "decoder") if cache else None

    for i in range(0, num_layers, moe_step):
        # encoder
        device_map_nllb_54b[f"model.encoder.layers.{i}"] = cuda
        device_map_nllb_54b[f"model.encoder.layers.{i+1}"] = cuda
        device_map_nllb_54b[f"model.encoder.layers.{i+2}"] = cuda

        device_map_nllb_54b[f"model.encoder.layers.{i+3}.self_attn"] = cuda
        device_map_nllb_54b[f"model.encoder.layers.{i+3}.attn_dropout"] = cuda
        device_map_nllb_54b[f"model.encoder.layers.{i+3}.self_attn_layer_norm"] = cuda
        device_map_nllb_54b[f"model.encoder.layers.{i+3}.ffn.router"] = cuda
        device_map_nllb_54b[f"model.encoder.layers.{i+3}.ffn.token_dropout"] = cuda
        for e in range(128):
            device_map_nllb_54b[f"model.encoder.layers.{i+3}.ffn.experts.expert_{e}"] = offload_device
        device_map_nllb_54b[f"model.encoder.layers.{i+3}.ff_layer_norm"] = cuda
        device_map_nllb_54b[f"model.encoder.layers.{i+3}.ff_dropout"] = cuda


        # decoder
        device_map_nllb_54b[f"model.decoder.layers.{i}"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+1}"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+2}"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+3}.self_attn"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+3}.attn_dropout"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+3}.self_attn_layer_norm"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+3}.activation_fn"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+3}.cross_attention"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+3}.cross_attention_layer_norm"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+3}.ffn.router"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+3}.ffn.token_dropout"] = cuda
        for e in range(128):
            device_map_nllb_54b[f"model.decoder.layers.{i+3}.ffn.experts.expert_{e}"] = offload_device

        if cache:
            for e in hot_experts[i//moe_step][:num_cached_experts]:
                device_map_nllb_54b[f"model.decoder.layers.{i+3}.ffn.experts.expert_{e}"] = cuda

        device_map_nllb_54b[f"model.decoder.layers.{i+3}.ff_layer_norm"] = cuda
        device_map_nllb_54b[f"model.decoder.layers.{i+3}.ff_dropout"] = cuda

    return device_map_nllb_54b
