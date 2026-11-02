import os
from datasets import load_dataset
from device_map import get_device_map_nllb_54b


MODEL_DIR = os.environ.get("MODEL_DIR", "")

MODELS = {
    "nllb-moe-54b" : os.path.join(MODEL_DIR, "nllb-moe-54b"),
}

ENCODER_DECODER_MODELS = ["nllb-moe-54b"]

INSTRUCT_MODELS = []

GET_DEVICE_MAP = {
    "nllb-moe-54b" : get_device_map_nllb_54b,
}

DATASETS = ["wmt"]
LANGUAGES = ["fr", "de"]


def load_eval_dataset(dataset_name, lang=None):
    if dataset_name == "wmt":
        return load_dataset("wmt/wmt14", f"{lang}-en", split="validation")['translation']
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
