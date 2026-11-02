import contextlib
import csv
import io
import json
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextvars import ContextVar
from unittest.mock import patch

import torch
from transformers import NllbMoeConfig
from transformers.modeling_outputs import BaseModelOutput

from device_map import get_device_map_nllb_54b
from transformers.models.nllb_moe.modeling_nllb_moe import NllbMoeDecoder, NllbMoeSparseMLP
from modelings.nllb import ExpertRouting, NllbMoeForConditionalGeneration, extend_decoder
from run import RESULT_FIELDS, _trim_generated_ids, _write_batch_result, parse_args
from specdec.decoding import DecodingConfig
from specdec.specdec import SpecDec


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cli_policy_and_legacy_temperature_for_every_method(self):
        with patch.dict(os.environ, {"MODEL_DIR": "/tmp/specmoe-test-checkpoints"}):
            for method in ("specdec", "cache", "nocache"):
                with self.subTest(method=method):
                    base = ["--method", method]
                    self.assertTrue(parse_args(base).decoding_config.greedy)
                    self.assertTrue(parse_args(base + ["-T", "0"]).decoding_config.greedy)
                    sampling = parse_args(base + ["--decoding", "sampling", "--temperature", "0.7",
                                                  "--top-k", "5", "--top-p", "0.9", "--seed", "42"])
                    self.assertEqual(sampling.decoding_config, DecodingConfig("sampling", 0.7, 5, 0.9))
                    self.assertEqual(sampling.seed, 42)
                    self.assertEqual(parse_args(base + ["-T", "0.8"]).decoding, "sampling")

    def test_invalid_configuration_is_rejected_before_loading(self):
        cases = [
            ["--decoding", "sampling", "-T", "0"],
            ["--decoding", "greedy", "--top-k", "5"],
            ["--decoding", "greedy", "-T", "1"],
            ["-T", "nan"], ["-T", "inf"], ["-T", "-1"],
            ["--top-k", "-1"], ["--top-p", "0"], ["--top-p", "nan"],
            ["--seed", "-1"], ["--num-fixed-experts", "0"],
            ["--num-cached-experts", "14"], ["--target", "llama4-scout"],
        ]
        with patch.dict(os.environ, {"MODEL_DIR": "/tmp/specmoe-test-checkpoints"}):
            for flags in cases:
                with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(["--method", "specdec"] + flags)

    def test_sampling_scores_match_hf_generation(self):
        torch.manual_seed(4)
        model_config = NllbMoeConfig(
            vocab_size=32, d_model=8, encoder_ffn_dim=16, decoder_ffn_dim=16,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
            decoder_attention_heads=2, encoder_sparse_step=2, decoder_sparse_step=2,
            dropout=0, attention_dropout=0, activation_dropout=0,
            decoder_start_token_id=2, bos_token_id=0, pad_token_id=1, eos_token_id=31,
        )
        model = NllbMoeForConditionalGeneration(model_config).eval()
        tokenizer = SimpleNamespace(eos_token_id=31, pad_token_id=1)
        ids = torch.tensor([[4, 5, 6], [7, 8, 9]])
        with torch.no_grad():
            raw_logits = model(input_ids=ids, decoder_input_ids=torch.full((2, 1), 2)).logits
            for settings in (DecodingConfig(), DecodingConfig("sampling", 0.7, 0, 1),
                             DecodingConfig("sampling", 0.8, 5, 0.75)):
                with self.subTest(settings=settings):
                    generated = model.generate(
                        ids, generation_config=settings.generation_config(model_config, tokenizer, 1),
                        return_dict_in_generate=True, output_scores=True,
                    )
                    target = generated.scores[0].log_softmax(-1)
                    actual = settings.log_probs(raw_logits)[:, 0]
                    self.assertTrue(torch.allclose(actual, target, atol=1e-6))
                    batched = settings.log_probs(raw_logits.expand(-1, 3, -1))
                    self.assertTrue(torch.allclose(batched[:, 2], target, atol=1e-6))

    def test_fresh_config_ignores_checkpoint_generation_defaults(self):
        model_config = SimpleNamespace(bos_token_id=0, decoder_start_token_id=2,
                                       num_beams=4, top_k=50, forced_eos_token_id=31)
        tokenizer = SimpleNamespace(eos_token_id=31, pad_token_id=1)
        cfg = DecodingConfig("sampling", 0.8, 0, 1).generation_config(model_config, tokenizer, 8, 30)
        self.assertEqual((cfg.num_beams, cfg.top_k, cfg.top_p), (1, 0, 1))
        self.assertIsNone(cfg.forced_eos_token_id)
        self.assertEqual(cfg.forced_bos_token_id, 30)

    def test_acceptance_and_resampling_preserve_target_distribution(self):
        torch.manual_seed(7)
        rows = 20000
        p = torch.tensor([0.1, 0.2, 0.7])
        q = torch.tensor([0.7, 0.2, 0.1])
        draft = torch.multinomial(q.expand(rows, -1), 1)
        px = p.log().expand(rows, 2, -1)
        qx = q.log().expand(rows, 1, -1)
        spec = SpecDec.__new__(SpecDec)
        accepted = spec.evaluate(draft, px, qx, torch.ones(rows, 1, dtype=torch.long))
        replacement = spec.resample(accepted, px, qx)
        first = torch.where(accepted.bool(), draft, replacement).flatten()
        observed = torch.bincount(first, minlength=3).float() / rows
        self.assertTrue(torch.allclose(observed, p, atol=0.01), observed)

    def test_csv_records_effective_settings_and_rejects_old_schema(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"MODEL_DIR": folder}):
            path = os.path.join(folder, "run.csv")
            args = parse_args(["--method", "cache", "--decoding", "sampling", "-T", "0.7",
                               "--top-k", "5", "--top-p", "0.9", "--seed", "42", "--output", path])
            _write_batch_result(args, 0, 2, 10)
            with open(path, newline="") as source:
                reader = csv.DictReader(source)
                self.assertEqual(reader.fieldnames, RESULT_FIELDS)
                row = next(reader)
            self.assertEqual([row[name] for name in ("decoding", "temperature", "top_k", "top_p", "seed")],
                             ["sampling", "0.7", "5", "0.9", "42"])
            self.assertEqual(row["num_cached_experts"], "13")
            self.assertEqual(row["num_fixed_experts"], "")
            Path(path).write_text("old,header\n")
            with self.assertRaises(ValueError):
                _write_batch_result(args, 1, 2, 10)

    def test_eos_padding_is_trimmed_without_dropping_generated_pad_tokens(self):
        rows = _trim_generated_ids(torch.tensor([[4, 9, 1], [1, 4, 5]]), 9)
        self.assertEqual(rows, [[4, 9], [1, 4, 5]])

    def test_cache_count_applies_only_to_decoder_experts(self):
        mapping = get_device_map_nllb_54b("cpu", cache=True, num_cached_experts=3)
        decoder = [value for key, value in mapping.items() if "decoder" in key and ".experts." in key]
        encoder = [value for key, value in mapping.items() if "encoder" in key and ".experts." in key]
        self.assertEqual(decoder.count(0), 6 * 3)
        self.assertEqual(encoder.count(0), 0)

    def test_larger_draft_initialization_uses_requested_count(self):
        config = NllbMoeConfig(d_model=8, num_experts=128)
        moe = ExpertRouting(NllbMoeSparseMLP(config, ffn_dim=16), ContextVar("draft", default=False), layer_idx=3)
        moe.init_fixed_expert(16)
        self.assertEqual(len(moe.fixed_expert), 16)
        self.assertEqual(len(set(moe.fixed_expert.tolist())), 16)

    def test_affinity_files_match_manifest_and_model_layout(self):
        root = Path(__file__).resolve().parents[1] / "affinity_tables"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual({item["decoder_layer"] for item in manifest["files"].values()}, set(range(3, 24, 4)))
        for name, item in manifest["files"].items():
            data = (root / name).read_bytes()
            self.assertEqual(len(data), item["size_bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            tensor = torch.load(root / name, weights_only=True, map_location="cpu")
            self.assertEqual(tensor.shape, (128, 128))
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertTrue(torch.equal(tensor, tensor.T))


class NllbBatchingTests(unittest.TestCase):
    def test_masked_shared_cache_matches_independent_compact_decoding(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        config = NllbMoeConfig(
            vocab_size=32, d_model=8, encoder_ffn_dim=16, decoder_ffn_dim=16,
            encoder_layers=1, decoder_layers=4, encoder_attention_heads=2,
            decoder_attention_heads=2, num_experts=4, encoder_sparse_step=4,
            decoder_sparse_step=4, max_position_embeddings=4, dropout=0,
            attention_dropout=0, activation_dropout=0, moe_token_dropout=0,
            second_expert_policy="all", moe_eval_capacity_token_fraction=1.0,
            decoder_start_token_id=2, bos_token_id=0, pad_token_id=1, eos_token_id=31,
        )
        decoder = NllbMoeDecoder(config).eval()
        extend_decoder(decoder)
        encoder_states = torch.randn(2, 4, 8)
        encoder_mask = torch.ones(2, 4, dtype=torch.long)
        queries = torch.tensor([[7, 8, 9], [10, 11, 12]])
        # Cover both trailing holes and holes between valid cached tokens.
        prefixes = (
            torch.tensor([[2, 3, 4, 5], [2, 6, 1, 1]]),
            torch.tensor([[2, 3, 4, 5, 6, 7], [2, 8, 1, 1, 9, 1]]),
        )
        with torch.no_grad():
            for prefix in prefixes:
                with self.subTest(prefix=prefix.tolist()):
                    prefix_mask = prefix.ne(config.pad_token_id).long()
                    prefill = decoder(
                        input_ids=prefix, attention_mask=prefix_mask,
                        encoder_hidden_states=encoder_states, encoder_attention_mask=encoder_mask,
                        use_cache=True,
                    )
                    shared_mask = torch.cat((prefix_mask, torch.ones_like(queries)), dim=1)
                    shared_inputs = dict(
                        attention_mask=shared_mask, past_key_values=prefill.past_key_values,
                        encoder_hidden_states=encoder_states, encoder_attention_mask=encoder_mask,
                        use_cache=True,
                    )
                    packed = decoder(input_ids=queries, **shared_inputs).last_hidden_state
                    references = []
                    for row in range(prefix.size(0)):
                        compact_prefix = prefix[row:row + 1, prefix_mask[row].bool()]
                        compact_prefill = decoder(
                            input_ids=compact_prefix, attention_mask=torch.ones_like(compact_prefix),
                            encoder_hidden_states=encoder_states[row:row + 1],
                            encoder_attention_mask=encoder_mask[row:row + 1], use_cache=True,
                        )
                        compact = decoder(
                            input_ids=queries[row:row + 1],
                            attention_mask=torch.ones(1, compact_prefix.size(1) + queries.size(1)),
                            past_key_values=compact_prefill.past_key_values,
                            encoder_hidden_states=encoder_states[row:row + 1],
                            encoder_attention_mask=encoder_mask[row:row + 1], use_cache=True,
                        ).last_hidden_state
                        references.append(compact)
                    torch.testing.assert_close(packed, torch.cat(references), rtol=1e-5, atol=1e-5)
                    embedded = decoder(
                        inputs_embeds=decoder.embed_tokens(queries), **shared_inputs
                    ).last_hidden_state
                    torch.testing.assert_close(embedded, packed, rtol=1e-5, atol=1e-5)
        self.assertGreater(decoder.embed_positions.weights.size(0), config.max_position_embeddings + 2)


class SpeculativePolicyTests(unittest.TestCase):
    def test_filtered_draft_and_verify_with_forced_bos_and_exact_budget(self):
        class Tokenizer:
            eos_token_id = 7
            pad_token = "<pad>"

            def __call__(self, prompts, **kwargs):
                ids = torch.tensor([[2, 3]] * len(prompts))
                return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

            def batch_decode(self, ids):
                return [str(row) for row in ids]

        class Model:
            config = SimpleNamespace(decoder_start_token_id=2)

            def get_encoder(self):
                return lambda ids, **kwargs: BaseModelOutput(last_hidden_state=ids.float().unsqueeze(-1))

            def forward(self, decoder_input_ids, past_key_values, fixed_gating, **kwargs):
                rows, steps = decoder_input_ids.shape
                logits = torch.full((rows, steps, 8), -100.0)
                logits[..., 4 if fixed_gating else 3] = 10
                previous = 0 if past_key_values is None else past_key_values[0][0].size(2)
                kv = torch.zeros(rows, 1, previous + steps, 1)
                cross = torch.zeros(rows, 1, 2, 1)
                return SimpleNamespace(logits=logits, past_key_values=((kv, kv.clone(), cross, cross.clone()),))

        spec = SpecDec.__new__(SpecDec)
        spec.target_model = Model()
        spec.tokenizer = Tokenizer()
        spec.is_encoder_decoder = True
        spec.is_instruct = False
        spec.tokens_per_step = 3
        with torch.no_grad(), patch.object(torch.Tensor, "cuda", lambda tensor, *a, **k: tensor):
            info = spec.speculative_decoding(
                ["a", "b"], input_length=2, max_tokens=6, forced_bos_token_id=1,
                decoding_config=DecodingConfig("sampling", 0.8, 1, 0.7),
            )
        self.assertEqual(info["output_ids"], [[1, 3, 3, 3], [1, 3, 3, 3]])


if __name__ == "__main__":
    unittest.main()
