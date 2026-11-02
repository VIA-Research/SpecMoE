"""Compare speculative and target greedy tokens using a real NLLB checkpoint."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import torch

from config import GET_DEVICE_MAP, MODELS
from patches import apply_patches
from run import _batches, _forced_bos, _trim_generated_ids, configure_logging, parse_args
from specdec.specdec import SpecDec


def compare_outputs(expected, actual, num_prompts):
    # zip alone silently ignores missing or extra output sequences.
    if len(expected) != num_prompts or len(actual) != num_prompts:
        raise ValueError(
            f"Output count differs: prompts={num_prompts}, "
            f"target={len(expected)}, specdec={len(actual)}"
        )
    differences = []
    for row, (target_ids, spec_ids) in enumerate(zip(expected, actual), 1):
        if target_ids != spec_ids:
            first = next((i for i, pair in enumerate(zip(target_ids, spec_ids))
                          if pair[0] != pair[1]), min(len(target_ids), len(spec_ids)))
            differences.append({"sequence": row, "token_index": first,
                                "target_ids": target_ids, "specdec_ids": spec_ids})
    return differences


def check_batches(args, report):
    # Validate dataset selection before loading the 54B checkpoint.
    batches = _batches(args)
    selected_samples = sum(map(len, batches))
    if selected_samples != args.num_samples:
        raise ValueError(f"Requested {args.num_samples} samples, but selected {selected_samples}; "
                         "use a complete number of batches within the validation split.")
    print(f"Loading NLLB-MoE 54B; checking {selected_samples} sequences", flush=True)
    spec = SpecDec(
        model_path=MODELS[args.target],
        device_map=GET_DEVICE_MAP[args.target](args.offload_device),
        is_encoder_decoder=True, gamma=args.gamma,
        offload_path=args.offload_dir if args.offload_device == "disk" else None,
    )
    spec.target_model.set_num_fixed_experts(args.num_fixed_experts)
    forced_bos = _forced_bos(spec.tokenizer, args)
    generation_config = args.decoding_config.generation_config(
        spec.target_model.config, spec.tokenizer, args.output_len, forced_bos,
    )
    with torch.no_grad():
        for batch_index, prompts in enumerate(batches, 1):
            print(f"Batch {batch_index}/{len(batches)}: target greedy", flush=True)
            inputs = spec.tokenizer(prompts, return_tensors="pt", padding=True,
                                    truncation=True, max_length=args.input_len)
            expected_tensor = spec.target_model.generate(
                input_ids=inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(),
                generation_config=generation_config,
            )[:, 1:]
            expected = _trim_generated_ids(expected_tensor, spec.tokenizer.eos_token_id)
            print(f"Batch {batch_index}/{len(batches)}: speculative greedy", flush=True)
            actual = spec.speculative_decoding(
                prompts, forced_bos_token_id=forced_bos, input_length=args.input_len,
                max_tokens=args.input_len + args.output_len, decoding_config=args.decoding_config,
            )["output_ids"]
            differences = compare_outputs(expected, actual, len(prompts))
            report["checked_sequences"] += len(prompts)
            report["mismatching_sequences"] += len(differences)
            report["batches"].append({"batch_index": batch_index,
                                      "target_ids": expected, "specdec_ids": actual,
                                      "differences": differences})
            for item in differences:
                print(f"Mismatch: batch={batch_index}, sequence={item['sequence']}, "
                      f"token_index={item['token_index']}", flush=True)
            print(f"Checked batch {batch_index}: {len(prompts)} sequences, "
                  f"{len(differences)} mismatches", flush=True)


def main(argv=None):
    args = parse_args(["--method", "specdec", "--decoding", "greedy"] +
                      (sys.argv[1:] if argv is None else argv))
    if args.method != "specdec" or not args.decoding_config.greedy:
        raise ValueError("This check requires --method=specdec and --decoding=greedy.")
    if args.target != "nllb-moe-54b":
        raise ValueError("This check supports only NLLB-MoE 54B.")
    if args.measure:
        raise ValueError("Use run.py for transfer profiling; this check compares output tokens.")
    configure_logging(args)
    torch.manual_seed(args.seed)
    apply_patches()
    root = Path(__file__).resolve().parents[1]
    started_at = datetime.now(timezone.utc)
    report_path = root / "results" / "validation" / (
        f"nllb_greedy_{started_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sources = ["scripts/check_greedy.py", "run.py", "config.py", "device_map.py",
               "specdec/specdec.py", "specdec/decoding.py", "modelings/nllb.py",
               "patches/__init__.py", "patches/models.py", "patches/accelerate_hooks.py"]
    checkpoint_config = Path(MODELS[args.target]) / "config.json"
    report = {
        "status": "running", "started_at": started_at.isoformat(),
        "checkpoint": str(Path(MODELS[args.target]).resolve()),
        "checkpoint_config_sha256": hashlib.sha256(checkpoint_config.read_bytes()).hexdigest(),
        "settings": {key: getattr(args, key) for key in (
            "target", "dataset", "lang", "offload_device", "batch_size", "num_samples",
            "input_len", "output_len", "gamma", "num_fixed_experts", "seed", "decoding")},
        "dtype": "float16", "python": sys.version.split()[0],
        "versions": {name: version(name) for name in (
            "torch", "transformers", "accelerate", "datasets", "sentencepiece")},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in sources},
        "checked_sequences": 0, "mismatching_sequences": 0, "batches": [],
    }
    print(f"Validation report: {report_path}", flush=True)
    started = time.monotonic()
    try:
        check_batches(args, report)
        report["status"] = "failed" if report["mismatching_sequences"] else "passed"
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved validation report: {report_path}", flush=True)
    print(f"Greedy comparison: {report['mismatching_sequences']} mismatching sequences "
          f"out of {report['checked_sequences']}", flush=True)
    return 1 if report["mismatching_sequences"] else 0


if __name__ == "__main__":
    sys.exit(main())
