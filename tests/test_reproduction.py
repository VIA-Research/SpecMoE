import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run import RESULT_FIELDS, _write_batch_result, parse_args
from scripts.run_experiments import (
    build_cases, build_command, load_config, main, summarize_case, write_summary,
)


class ReproductionTests(unittest.TestCase):
    def cases(self, **overrides):
        return build_cases(load_config(overrides=overrides))

    def test_configurations_generate_valid_commands(self):
        configurations = [({}, 18), ({"measure": "movement"}, 18),
                          ({"methods": ["specdec"], "batch_sizes": [256],
                            "num_fixed_experts": [2, 4, 8, 16]}, 4),
                          ({"methods": ["specdec"], "batch_sizes": [128],
                            "gamma": [10, 12, 15, 17, 20]}, 5),
                          ({"offload_device": "disk", "input_len": 64, "output_len": 32}, 18)]
        with patch.dict(os.environ, {"MODEL_DIR": "/tmp/models", "OFFLOAD_DIR": "/tmp/offload"}):
            for config, count in configurations:
                cases = self.cases(**config)
                self.assertEqual(len(cases), count)
                self.assertEqual(len({case["name"] for case in cases}), count)
                for case in cases:
                    with self.subTest(case=case["name"]):
                        args = parse_args(build_command(case, "/tmp/specmoe-unused.csv")[3:])
                        self.assertEqual(args.num_samples, 10 * case["batch_size"])
                        self.assertTrue(args.decoding_config.greedy)

    def test_method_specific_sweeps_do_not_duplicate_baselines(self):
        cases = self.cases(batch_sizes=[1, 4], gamma=[5, 10], num_fixed_experts=[2, 4],
                           num_cached_experts=[0, 13])
        counts = {method: sum(case["method"] == method for case in cases)
                  for method in ("nocache", "cache", "specdec")}
        self.assertEqual(counts, {"nocache": 2, "cache": 4, "specdec": 8})

    def test_config_file_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "custom.json"
            path.write_text(json.dumps({"methods": ["specdec"], "batch_sizes": [3], "gamma": [7]}))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--config", str(path), "--gamma", "9", "11", "--print-config"]), 0)
            config = json.loads(output.getvalue())
            self.assertEqual(config["gamma"], [9, 11])
            self.assertEqual(config["batch_sizes"], [3])
            self.assertEqual(config["methods"], ["specdec"])
            self.assertEqual(config["input_len"], 128)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_invalid_config_does_not_reach_model_loading(self):
        for overrides in ({"gamma": []}, {"batch_sizes": [0]}, {"num_fixed_experts": [129]},
                          {"num_cached_experts": [14]}, {"method": ["cache"]},
                          {"measure": "invalid"}, {"num_batches": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_config(overrides=overrides)

    def test_sampling_settings_and_language_reach_every_method(self):
        cases = self.cases(batch_sizes=[2], lang="de", decoding="sampling", temperature=0.8, top_k=50, top_p=0.9)
        with patch.dict(os.environ, {"MODEL_DIR": "/tmp/models"}):
            for case in cases:
                args = parse_args(build_command(case, "/tmp/specmoe-unused.csv")[3:])
                self.assertEqual((args.lang, args.decoding, args.temperature, args.top_k, args.top_p),
                                 ("de", "sampling", 0.8, 50, 0.9))

    def write_case(self, folder, case, seconds, tokens, transfers=None):
        path = folder / case["name"] / "attempt_001" / "batches.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            for index, (elapsed, count) in enumerate(zip(seconds, tokens), 1):
                row = {name: case.get(name, "") for name in RESULT_FIELDS}
                row.update(target="nllb-moe-54b", dataset="wmt", lang="fr", decoding="greedy",
                           measure=case["measure"] or "", batch_index=index,
                           latency_s=elapsed, output_tokens=count, generated_per_step="")
                if transfers is not None:
                    row["transfer_bytes"] = transfers[index - 1]
                writer.writerow(row)
        (path.parent.parent / "state.json").write_text(json.dumps({"status": "passed", "attempt": "attempt_001"}))
        return path

    def test_weighted_throughput_and_normalization_use_same_scope(self):
        cases = self.cases(methods=["nocache", "specdec"], batch_sizes=[1], num_batches=2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_case(root, cases[0], [2, 8], [20, 40])
            self.write_case(root, cases[1], [1, 4], [20, 40])
            write_summary(root, {"cases": cases})
            with (root / "summary.csv").open() as file:
                rows = list(csv.DictReader(file))
        by_key = {(row["method"], row["scope"]): row for row in rows}
        self.assertEqual(float(by_key["nocache", "all"]["tokens_per_s"]), 6)
        self.assertEqual(float(by_key["nocache", "after_first"]["tokens_per_s"]), 5)
        self.assertEqual(float(by_key["specdec", "all"]["normalized_latency"]), 0.5)
        self.assertEqual(float(by_key["specdec", "after_first"]["normalized_throughput"]), 2)

    def test_transfer_summary_does_not_normalize_instrumented_throughput(self):
        cases = self.cases(methods=["nocache", "specdec"], batch_sizes=[1], num_batches=1, measure="movement")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_case(root, cases[0], [4], [20], [1000])
            self.write_case(root, cases[1], [2], [20], [250])
            write_summary(root, {"cases": cases})
            with (root / "summary.csv").open() as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(float(rows[1]["transfer_ratio_vs_nocache"]), 0.25)
        self.assertEqual(float(rows[1]["transfer_bytes_per_output_token"]), 12.5)
        self.assertEqual(rows[1]["normalized_throughput"], "")

    def test_incomplete_csv_and_wrong_expert_count_are_rejected(self):
        case = self.cases(methods=["specdec"], batch_sizes=[1], num_batches=2, num_fixed_experts=[2])[0]
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_case(Path(folder), case, [1], [20])
            with self.assertRaisesRegex(ValueError, "expected 2 batches"):
                summarize_case(case, path)
            case = {**case, "num_batches": 1, "num_fixed_experts": 16}
            with self.assertRaisesRegex(ValueError, "num_fixed_experts"):
                summarize_case(case, path)

    def test_measurement_counts_are_saved_in_batch_csv(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"MODEL_DIR": "/tmp/models"}):
            for mode in (None, "movement", "overlap"):
                path = Path(folder) / f"{mode}.csv"
                cli = ["--method", "specdec", "--output", str(path)]
                if mode:
                    cli += ["--measure", mode]
                args = parse_args(cli)
                with patch("patches.measurement.report", return_value={"bytes": 4096, "ms": 12.5, "count": 4}):
                    _write_batch_result(args, 0, 2, 20)
                with path.open() as file:
                    row = next(csv.DictReader(file))
                self.assertEqual(row["transfer_bytes"], "4096" if mode == "movement" else "")
                self.assertEqual(row["transfer_ms"], "12.5" if mode == "overlap" else "")
                self.assertEqual(row["transfer_count"], "4" if mode else "")


if __name__ == "__main__":
    unittest.main()
