"""Run configurable inference sweeps and aggregate their results."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "nllb.json"
METHODS = ["nocache", "cache", "specdec"]
CONFIG_FIELDS = {
    "target", "dataset", "lang", "methods", "batch_sizes", "input_len", "output_len",
    "offload_device", "gamma", "num_fixed_experts", "num_cached_experts", "num_batches",
    "decoding", "temperature", "top_k", "top_p", "seed", "measure",
}
MATCH_FIELDS = (
    "target", "dataset", "lang", "batch_size", "input_len", "output_len",
    "offload_device", "num_samples", "seed", "decoding", "temperature", "top_k", "top_p", "measure",
)
SUMMARY_FIELDS = [
    "case", "scope", "method", *MATCH_FIELDS, "gamma", "num_fixed_experts",
    "num_cached_experts", "batches",
    "output_tokens", "latency_s", "mean_batch_latency_s", "tokens_per_s",
    "mean_batch_generated_per_step", "transfer_bytes", "mean_batch_transfer_bytes",
    "transfer_bytes_per_output_token", "normalized_latency", "normalized_throughput",
    "transfer_ratio_vs_nocache",
]


def load_config(path=DEFAULT_CONFIG, overrides=None):
    config = json.loads(DEFAULT_CONFIG.read_text())
    if Path(path).resolve() != DEFAULT_CONFIG.resolve():
        extra = json.loads(Path(path).read_text())
        if not isinstance(extra, dict):
            raise ValueError("Configuration must be a JSON object.")
        config.update(extra)
    if overrides:
        config.update(overrides)
    unknown = config.keys() - CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
    for name in ("methods", "batch_sizes", "gamma", "num_fixed_experts", "num_cached_experts"):
        values = config[name] if isinstance(config[name], list) else [config[name]]
        if not values:
            raise ValueError(f"{name} must not be empty.")
        if name == "methods":
            if any(value not in METHODS for value in values):
                raise ValueError(f"methods must be selected from {METHODS}.")
        else:
            minimum = 0 if name == "num_cached_experts" else 1
            maximum = {"num_fixed_experts": 128, "num_cached_experts": 13}.get(name)
            if any(type(value) is not int or value < minimum or
                   (maximum is not None and value > maximum) for value in values):
                raise ValueError(f"Invalid {name}: {values}")
        config[name] = list(dict.fromkeys(values))
    for name in ("input_len", "output_len", "num_batches"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for name, choices in (("target", ["nllb-moe-54b"]), ("dataset", ["wmt"]),
                          ("lang", ["fr", "de"]), ("offload_device", ["cpu", "disk"]),
                          ("decoding", ["greedy", "sampling"]), ("measure", [None, "movement", "overlap"])):
        if config[name] not in choices:
            raise ValueError(f"{name} must be one of {choices}.")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63).")
    for name in ("temperature", "top_p"):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
        config[name] = float(config[name])
    if type(config["top_k"]) is not int or config["top_k"] < 0 or config["top_p"] > 1:
        raise ValueError("top_k must be a nonnegative integer and top_p must be in (0, 1].")
    if config["decoding"] == "greedy" and (config["top_k"] or config["top_p"] != 1):
        raise ValueError("top_k and top_p filtering require sampling decoding.")
    return config


def build_cases(config):
    cases = []
    shared = {name: config[name] for name in MATCH_FIELDS if name not in ("batch_size", "num_samples")}
    shared["temperature"] = config["temperature"] if config["decoding"] == "sampling" else 0.0
    for method in config["methods"]:
        gammas = config["gamma"] if method == "specdec" else [None]
        fixed_counts = config["num_fixed_experts"] if method == "specdec" else [None]
        cached_counts = config["num_cached_experts"] if method == "cache" else [None]
        for batch, gamma, fixed, cached in itertools.product(config["batch_sizes"], gammas, fixed_counts, cached_counts):
            name = f"{method}_{config['lang']}_{config['offload_device']}_b{batch}_i{config['input_len']}_o{config['output_len']}"
            if method == "specdec":
                name += f"_g{gamma}_d{fixed}"
            elif method == "cache":
                name += f"_c{cached}"
            name += f"_{config['decoding']}_{config['measure'] or 'plain'}"
            cases.append({**shared, "name": name, "method": method, "batch_size": batch,
                          "gamma": gamma, "num_fixed_experts": fixed, "num_cached_experts": cached,
                          "num_samples": batch * config["num_batches"], "num_batches": config["num_batches"]})
    return cases


def build_command(case, output, python=None):
    command = [python or sys.executable, "-u", str(ROOT / "run.py")]
    for name in ("target", "dataset", "lang", "decoding", "method", "batch_size",
                 "input_len", "output_len", "offload_device", "num_samples", "seed"):
        command += ["--" + name.replace("_", "-"), str(case[name])]
    if case["method"] == "specdec":
        command += ["--gamma", str(case["gamma"]), "--num-fixed-experts", str(case["num_fixed_experts"])]
    elif case["method"] == "cache":
        command += ["--num-cached-experts", str(case["num_cached_experts"])]
    if case["measure"]:
        command += ["--measure", case["measure"]]
    if case["decoding"] == "sampling":
        command += ["--temperature", str(case["temperature"]), "--top-k", str(case["top_k"]),
                    "--top-p", str(case["top_p"])]
    return command + ["--output", str(output)]


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_hashes():
    paths = [ROOT / name for name in ("run.py", "config.py", "device_map.py", "requirements.txt", "run.sh")]
    for folder, pattern in (("scripts", "*.py"), ("specdec", "*.py"), ("patches", "*.py"),
                            ("modelings", "*.py"), ("hot_experts", "*.json"),
                            ("affinity_tables", "*.pt")):
        paths += sorted((ROOT / folder).glob(pattern))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def summarize_case(case, csv_path):
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != case["num_batches"]:
        raise ValueError(f"{case['name']}: expected {case['num_batches']} batches, got {len(rows)}")
    for index, row in enumerate(rows, 1):
        if int(row["batch_index"]) != index:
            raise ValueError(f"{case['name']}: duplicate or missing batch index")
        for name in ("method", *MATCH_FIELDS):
            if row[name] != str(case[name] if case[name] is not None else ""):
                raise ValueError(f"{case['name']}: CSV setting differs: {name}")
        keys = ("gamma", "num_fixed_experts") if case["method"] == "specdec" else (
            ("num_cached_experts",) if case["method"] == "cache" else ())
        for name in keys:
            if row[name] != str(case[name]):
                raise ValueError(f"{case['name']}: CSV setting differs: {name}")
        if not math.isfinite(float(row["latency_s"])) or float(row["latency_s"]) <= 0 or int(row["output_tokens"]) <= 0:
            raise ValueError(f"{case['name']}: invalid latency or token count")
    summaries = []
    for scope, selected in (("all", rows), ("after_first", rows[1:])):
        if not selected:
            continue
        seconds = sum(float(row["latency_s"]) for row in selected)
        tokens = sum(int(row["output_tokens"]) for row in selected)
        generated = [float(row["generated_per_step"]) for row in selected if row["generated_per_step"]]
        movement = case["measure"] == "movement"
        total_bytes = sum(int(row["transfer_bytes"]) for row in selected) if movement else None
        summary = {name: case[name] for name in ("method", *MATCH_FIELDS)}
        summary.update(
            case=case["name"], scope=scope, measure=case["measure"] or "",
            gamma=case["gamma"] if case["method"] == "specdec" else "",
            num_fixed_experts=case["num_fixed_experts"] if case["method"] == "specdec" else "",
            num_cached_experts=case["num_cached_experts"] if case["method"] == "cache" else "",
            batches=len(selected), output_tokens=tokens, latency_s=seconds,
            mean_batch_latency_s=seconds / len(selected), tokens_per_s=tokens / seconds,
            mean_batch_generated_per_step=sum(generated) / len(generated) if generated else "",
            transfer_bytes=total_bytes if movement else "",
            mean_batch_transfer_bytes=total_bytes / len(selected) if movement else "",
            transfer_bytes_per_output_token=total_bytes / tokens if movement else "",
            normalized_latency="", normalized_throughput="", transfer_ratio_vs_nocache="",
        )
        summaries.append(summary)
    return summaries


def write_summary(directory, manifest):
    rows = []
    for case in manifest["cases"]:
        state_path = directory / case["name"] / "state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        if state["status"] == "passed":
            rows += summarize_case(case, directory / case["name"] / state["attempt"] / "batches.csv")
    def reference_key(row):
        return (row["scope"], *(row[name] for name in MATCH_FIELDS))

    references = {reference_key(row): row
                  for row in rows if row["method"] == "nocache"}
    for row in rows:
        baseline = references.get(reference_key(row))
        if baseline is None:
            continue
        if not row["measure"]:
            row["normalized_latency"] = row["mean_batch_latency_s"] / baseline["mean_batch_latency_s"]
            row["normalized_throughput"] = row["tokens_per_s"] / baseline["tokens_per_s"]
        elif baseline["transfer_bytes"]:
            row["transfer_ratio_vs_nocache"] = row["transfer_bytes"] / baseline["transfer_bytes"]
    with (directory / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_cases(directory, manifest):
    for index, case in enumerate(manifest["cases"], 1):
        case_dir = directory / case["name"]
        case_dir.mkdir(exist_ok=True)
        state_path = case_dir / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["status"] == "passed":
                summarize_case(case, case_dir / state["attempt"] / "batches.csv")
                print(f"[{index}/{len(manifest['cases'])}] Already complete: {case['name']}", flush=True)
                continue
        attempt = f"attempt_{len(list(case_dir.glob('attempt_*'))) + 1:03d}"
        attempt_dir = case_dir / attempt
        attempt_dir.mkdir()
        command = build_command(case, attempt_dir / "batches.csv")
        save_json(attempt_dir / "command.json", command)
        state = {"status": "running", "attempt": attempt}
        save_json(state_path, state)
        print(f"[{index}/{len(manifest['cases'])}] {case['name']} -> {attempt_dir / 'run.log'}", flush=True)
        try:
            with (attempt_dir / "run.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise
            if code:
                raise RuntimeError(f"exit code {code}; see {attempt_dir / 'run.log'}")
            summarize_case(case, attempt_dir / "batches.csv")
            state["status"] = "passed"
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save_json(state_path, state)
            write_summary(directory, manifest)
    print(f"Completed {len(manifest['cases'])} runs. Summary: {directory / 'summary.csv'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON config; missing fields use configs/nllb.json defaults.")
    parser.add_argument("--method", dest="methods", nargs="+", choices=METHODS)
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--gamma", nargs="+", type=int, help="Draft token counts to sweep for specdec.")
    parser.add_argument("--num-fixed-experts", nargs="+", type=int, help="Draft expert counts to sweep for specdec.")
    parser.add_argument("--num-cached-experts", nargs="+", type=int, help="Cached expert counts to sweep for cache.")
    parser.add_argument("--target", choices=["nllb-moe-54b"])
    parser.add_argument("--dataset", choices=["wmt"])
    parser.add_argument("--lang", choices=["fr", "de"])
    parser.add_argument("--offload-device", choices=["cpu", "disk"])
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--num-batches", type=int, help="Same number of batches for every method.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--decoding", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--measure", choices=["none", "movement", "overlap"])
    parser.add_argument("--output-dir", type=Path, help="New output directory; default is timestamped.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved configuration without executing.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without loading models or writing files.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path, help="Resume the original manifest, retrying incomplete cases.")
    mode.add_argument("--summarize", type=Path, help="Rebuild summary.csv from completed cases.")
    args = parser.parse_args(argv)
    overrides = {name: value for name, value in vars(args).items() if name in CONFIG_FIELDS and value is not None}
    if overrides.get("measure") == "none":
        overrides["measure"] = None
    if args.resume or args.summarize:
        if args.config or overrides or args.output_dir:
            parser.error("Resume and summarize use the saved configuration; omit config overrides and --output-dir.")
        directory = (args.resume or args.summarize).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_version") != 2:
            parser.error("This manifest uses an older format; start a new run with the current config format.")
        if args.summarize:
            write_summary(directory, manifest)
            print(directory / "summary.csv")
            return 0
        if not args.print_config and manifest["source_sha256"] != source_hashes():
            parser.error("Source or affinity files changed; start a new output directory to avoid mixing results.")
        print("Resuming the saved configuration.", file=sys.stderr)
    else:
        try:
            config = load_config(args.config or DEFAULT_CONFIG, overrides)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = (args.output_dir or ROOT / "results" / "reproduction" / stamp).resolve()
        manifest = {"schema_version": 2, "config": config,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "cases": build_cases(config), "source_sha256": source_hashes(),
                    "sampling_policy": "First batch_size * num_batches validation rows in original order; FP16 weights.",
                    "aggregation": "All batches and after-first-batch summaries are kept separately; throughput=sum(tokens)/sum(seconds)."}
    if args.print_config:
        print(json.dumps(manifest["config"], indent=2))
        return 0
    if args.dry_run:
        for case in manifest["cases"]:
            print(shlex.join(build_command(case, directory / case["name"] / "attempt_001" / "batches.csv")))
        print(f"{len(manifest['cases'])} cases; output directory: {directory}")
        return 0
    checkpoint = Path(os.environ.get("MODEL_DIR", "")) / manifest["config"]["target"] / "config.json"
    if not os.environ.get("MODEL_DIR") or not checkpoint.is_file():
        parser.error("Set MODEL_DIR to the checkpoint parent directory containing nllb-moe-54b/config.json.")
    if any(case["offload_device"] == "disk" for case in manifest["cases"]) and not os.environ.get("OFFLOAD_DIR"):
        parser.error("Disk offloading requires OFFLOAD_DIR.")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if args.resume:
        if manifest["checkpoint_config_sha256"] != checkpoint_hash:
            parser.error("Checkpoint config differs from the saved manifest.")
        if manifest["checkpoint"] != str(checkpoint.parent.resolve()):
            parser.error("Checkpoint path differs from the saved manifest.")
        if manifest["offload_dir"] != os.environ.get("OFFLOAD_DIR"):
            parser.error("OFFLOAD_DIR differs from the saved manifest.")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        manifest.update(checkpoint_config_sha256=checkpoint_hash,
                        checkpoint=str(checkpoint.parent.resolve()), python=sys.executable,
                        offload_dir=os.environ.get("OFFLOAD_DIR"))
        save_json(directory / "manifest.json", manifest)
    print(f"Results: {directory}", flush=True)
    run_cases(directory, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
