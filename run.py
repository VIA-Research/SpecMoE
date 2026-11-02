import argparse
import csv
from datetime import datetime
import logging
import os
import sys
import warnings

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DATASETS, GET_DEVICE_MAP, LANGUAGES, MODELS, load_eval_dataset
from patches import apply_patches
from specdec.decoding import DecodingConfig
from specdec.specdec import SpecDec
from specdec.utils import Timer


logger = logging.getLogger("specmoe")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

METHODS = ["specdec", "cache", "nocache"]
RESULT_FIELDS = [
    "timestamp", "method", "target", "dataset", "lang", "offload_device",
    "batch_size", "input_len", "output_len", "num_samples", "gamma",
    "num_fixed_experts", "num_cached_experts", "decoding", "temperature",
    "top_k", "top_p", "seed", "measure", "batch_index", "latency_s",
    "output_tokens", "tokens_per_s", "generated_per_step",
    "transfer_bytes", "transfer_ms", "transfer_count",
]


def print_banner(args):
    parts = [f"method={args.method}", f"target={args.target}",
             f"batch={args.batch_size}", f"offload={args.offload_device}",
             f"decoding={args.decoding}", f"seed={args.seed}"]
    if args.decoding == "sampling":
        parts.extend([f"T={args.temperature}", f"top_k={args.top_k}", f"top_p={args.top_p}"])
    if args.method == "specdec":
        parts.extend([f"gamma={args.gamma}", f"fixed_experts={args.num_fixed_experts}"])
    print(f"==== SpecMoE run: {' / '.join(parts)} ====", flush=True)


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _log_measurement(mode):
    from patches import measurement
    stats = measurement.report()
    if mode == "movement":
        logger.info("  data movement: %.1f MB over %d transfers",
                    stats["bytes"] / 2**20, stats["count"])
    else:
        logger.info("  transfer time: %.1f ms over %d transfers", stats["ms"], stats["count"])


def _check_output_schema(path):
    if os.path.isfile(path) and os.path.getsize(path):
        with open(path, newline="") as existing:
            if next(csv.reader(existing), None) != RESULT_FIELDS:
                raise ValueError(f"CSV schema differs at {path}; choose a new --output path.")


def _write_batch_result(args, batch_idx, latency, output_tokens, generated_per_step=None):
    _check_output_schema(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    needs_header = not os.path.isfile(args.output) or os.path.getsize(args.output) == 0
    stats = {}
    if args.measure:
        from patches import measurement
        stats = measurement.report()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": args.method, "target": args.target, "dataset": args.dataset,
        "lang": args.lang, "offload_device": args.offload_device,
        "batch_size": args.batch_size, "input_len": args.input_len,
        "output_len": args.output_len, "num_samples": args.num_samples,
        "gamma": args.gamma if args.method == "specdec" else "",
        "num_fixed_experts": args.num_fixed_experts if args.method == "specdec" else "",
        "num_cached_experts": args.num_cached_experts if args.method == "cache" else "",
        "decoding": args.decoding, "temperature": args.decoding_config.effective_temperature,
        "top_k": args.top_k, "top_p": args.top_p, "seed": args.seed,
        "measure": args.measure or "", "batch_index": batch_idx + 1,
        "latency_s": f"{latency:.6f}", "output_tokens": output_tokens,
        "tokens_per_s": f"{output_tokens / latency:.6f}",
        "generated_per_step": f"{generated_per_step:.6f}" if generated_per_step is not None else "",
        "transfer_bytes": stats.get("bytes", "") if args.measure == "movement" else "",
        "transfer_ms": stats.get("ms", "") if args.measure == "overlap" else "",
        "transfer_count": stats.get("count", ""),
    }
    with open(args.output, "a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _trim_generated_ids(generated_ids, eos_token_id):
    rows = generated_ids.tolist()
    if eos_token_id is not None:
        rows = [row[:row.index(eos_token_id) + 1] if eos_token_id in row else row for row in rows]
    return rows


def _print_throughput_summary(latencies, output_tokens, generated_per_step=None):
    total_time = sum(latencies)
    total_tokens = sum(output_tokens)
    print(f"[{_ts()}] Total latency    : {total_time:.3f}s")
    print(f"[{_ts()}] Output tokens    : {total_tokens}")
    print(f"[{_ts()}] Token throughput : {total_tokens / total_time:.6f} tok/s")
    if generated_per_step:
        print(f"[{_ts()}] Avg gen/step     : {sum(generated_per_step) / len(generated_per_step):.2f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SpecMoE: NLLB-MoE inference with CPU/disk expert offloading on WMT14 En->X.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", required=True, choices=METHODS,
                        help="specdec = self-assisted draft; cache = pinned hot experts; nocache = all experts offloaded.")
    parser.add_argument("--target", default="nllb-moe-54b", choices=MODELS, help="Model to run.")
    parser.add_argument("--offload-device", default="cpu", choices=["cpu", "disk"],
                        help="Storage for experts offloaded from the GPU.")
    parser.add_argument("--dataset", default="wmt", choices=DATASETS, help="Evaluation dataset.")
    parser.add_argument("--lang", default="fr", choices=LANGUAGES, help="Translation target language.")
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("-n", "--input-len", type=int, default=128, help="Maximum input tokens.")
    parser.add_argument("--output-len", type=int, default=32, help="Maximum generated tokens per sequence.")
    parser.add_argument("-g", "--gamma", type=int, default=10, help="Draft tokens proposed per verification.")
    parser.add_argument("-D", "--num-fixed-experts", type=int, default=4,
                        help="specdec: draft experts pinned per decoder MoE layer (1..128).")
    parser.add_argument("--num-cached-experts", type=int, default=13,
                        help="cache: hot experts pinned per decoder MoE layer (0..13).")
    parser.add_argument("--decoding", choices=["greedy", "sampling"], default=None,
                        help="Shared decoding policy. Defaults to greedy; explicit -T > 0 selects sampling.")
    parser.add_argument("-T", "--temperature", type=float, default=None,
                        help="Sampling temperature (default 1). Legacy -T 0 selects greedy.")
    parser.add_argument("--top-k", type=int, default=0, help="Sampling vocabulary truncation; 0 disables it.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling nucleus mass; 1 disables it.")
    parser.add_argument("--seed", type=int, default=240725, help="PyTorch random seed for every method.")
    parser.add_argument("--num-samples", type=int, default=None, help="Defaults to batch_size * 10.")
    parser.add_argument("-o", "--output", default=None, help="CSV path; default includes method, language and decoding policy.")
    parser.add_argument("--measure", choices=["movement", "overlap"], default=None,
                        help="Per-batch transfer bytes or synchronized transfer wall time.")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("-v", "--verbose", action="store_true", help="Also log generated text.")
    log_group.add_argument("-q", "--quiet", action="store_true", help="Print only banner and final summary.")
    args = parser.parse_args(argv)

    if not os.environ.get("MODEL_DIR"):
        parser.error("MODEL_DIR must point at the directory containing nllb-moe-54b/.")
    if args.num_samples is None:
        args.num_samples = args.batch_size * 10
    for name in ("batch_size", "input_len", "output_len", "gamma", "num_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero.")
    if args.num_samples < args.batch_size:
        parser.error("--num-samples must be at least --batch-size.")
    if not 1 <= args.num_fixed_experts <= 128:
        parser.error("--num-fixed-experts must be in [1, 128].")
    if not 0 <= args.num_cached_experts <= 13:
        parser.error("--num-cached-experts must be in [0, 13], the provided hot-expert list size.")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2**63).")

    if args.decoding is None:
        args.decoding = "sampling" if args.temperature is not None and args.temperature > 0 else "greedy"
    if args.temperature == 0:
        if args.decoding == "sampling":
            parser.error("--decoding=sampling requires a positive temperature.")
        args.temperature = 1.0
    elif args.temperature is None:
        args.temperature = 1.0
    elif args.decoding == "greedy" and args.temperature > 0:
        parser.error("Use --decoding=sampling with a positive --temperature, or -T 0 for greedy.")
    try:
        args.decoding_config = DecodingConfig(args.decoding, args.temperature, args.top_k, args.top_p)
    except ValueError as error:
        parser.error(str(error))

    args.offload_dir = os.environ.get("OFFLOAD_DIR")
    if args.offload_device == "disk" and not args.offload_dir:
        parser.error("--offload-device=disk requires a writable OFFLOAD_DIR scratch directory.")
    if args.output is None:
        args.output = (f"{args.target}_{args.method}_{args.lang}_{args.offload_device}_"
                       f"{args.decoding}_b{args.batch_size}_g{args.gamma}.csv")
    try:
        _check_output_schema(args.output)
    except ValueError as error:
        parser.error(str(error))
    return args


def _batches(args):
    dataset = load_eval_dataset(args.dataset, lang=args.lang)[:args.num_samples]
    num_batches = len(dataset) // args.batch_size
    if not num_batches:
        raise ValueError("The dataset does not contain a complete batch for this configuration.")
    if len(dataset) % args.batch_size:
        logger.warning("Skipping %d samples in the incomplete final batch.", len(dataset) % args.batch_size)
    return [[item["en"] for item in dataset[i * args.batch_size:(i + 1) * args.batch_size]]
            for i in range(num_batches)]


def _forced_bos(tokenizer, args):
    lang_code = "fra_Latn" if args.lang == "fr" else "deu_Latn"
    return tokenizer.convert_tokens_to_ids(lang_code)


def _reset_measurement(args):
    if args.measure:
        from patches import measurement
        measurement.reset()


def _record_batch(args, index, total_batches, latency, output_info, latencies, token_counts):
    output_tokens = sum(len(row) for row in output_info["output_ids"])
    latencies.append(latency)
    token_counts.append(output_tokens)
    generated_per_step = output_info.get("num_generated")
    _write_batch_result(args, index, latency, output_tokens, generated_per_step)
    logger.info("batch %d/%d: %.3fs | %d output tokens | %.1f tok/s",
                index + 1, total_batches, latency, output_tokens, output_tokens / latency)
    if generated_per_step is not None:
        logger.info("  generated per step: %.2f", generated_per_step)
    if args.measure:
        _log_measurement(args.measure)
    logger.debug("output:\n%s", output_info["output_text"])


def run_specdec(args):
    device_map = GET_DEVICE_MAP[args.target](args.offload_device)
    spec = SpecDec(model_path=MODELS[args.target], device_map=device_map,
                   is_encoder_decoder=True, gamma=args.gamma,
                   offload_path=args.offload_dir if args.offload_device == "disk" else None)
    spec.target_model.set_num_fixed_experts(args.num_fixed_experts)
    batches = _batches(args)
    forced_bos = _forced_bos(spec.tokenizer, args)
    latencies, token_counts, generated_per_step = [], [], []
    print_banner(args)
    for index, prompts in enumerate(batches):
        _reset_measurement(args)
        with Timer() as timer:
            info = spec.speculative_decoding(
                prompts, forced_bos_token_id=forced_bos, input_length=args.input_len,
                max_tokens=args.input_len + args.output_len, decoding_config=args.decoding_config,
            )
        _record_batch(args, index, len(batches), timer.elapsed(), info, latencies, token_counts)
        generated_per_step.append(info["num_generated"])
    _print_throughput_summary(latencies, token_counts, generated_per_step)


def run_baseline(args):
    device_map = GET_DEVICE_MAP[args.target](
        args.offload_device, cache=args.method == "cache", num_cached_experts=args.num_cached_experts,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.target])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODELS[args.target], device_map=device_map, torch_dtype=torch.float16,
        trust_remote_code=False,
        offload_folder=args.offload_dir if args.offload_device == "disk" else None,
    )
    generation_config = args.decoding_config.generation_config(
        model.config, tokenizer, args.output_len, forced_bos_token_id=_forced_bos(tokenizer, args),
    )
    batches = _batches(args)
    latencies, token_counts = [], []
    print_banner(args)
    for index, prompts in enumerate(batches):
        _reset_measurement(args)
        # Both methods time tokenization, input transfer, encoder, decoding,
        # and conversion back to text.
        with Timer() as timer:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=args.input_len)
            generated_ids = model.generate(
                input_ids=inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(),
                generation_config=generation_config,
            )[:, 1:]
            output_ids = _trim_generated_ids(generated_ids, tokenizer.eos_token_id)
            info = {"output_ids": output_ids, "output_text": tokenizer.batch_decode(output_ids)}
        _record_batch(args, index, len(batches), timer.elapsed(), info, latencies, token_counts)
    _print_throughput_summary(latencies, token_counts)


def configure_logging(args):
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    transformers.logging.set_verbosity_error()


def main(args):
    torch.manual_seed(args.seed)
    apply_patches(measurement_mode=args.measure)
    with torch.no_grad():
        if args.method == "specdec":
            run_specdec(args)
        else:
            run_baseline(args)


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args)
    main(args)
