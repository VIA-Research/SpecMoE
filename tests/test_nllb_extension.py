from contextvars import ContextVar
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import NllbMoeConfig
from transformers.models.nllb_moe import modeling_nllb_moe as upstream

from modelings.nllb import ExpertRouting, NllbMoeForConditionalGeneration


def tiny_config():
    return NllbMoeConfig(
        vocab_size=32, d_model=8, encoder_ffn_dim=16, decoder_ffn_dim=16,
        encoder_layers=4, decoder_layers=4, encoder_attention_heads=2,
        decoder_attention_heads=2, num_experts=4, encoder_sparse_step=4,
        decoder_sparse_step=4, dropout=0, attention_dropout=0,
        activation_dropout=0, moe_token_dropout=0.2,
        second_expert_policy="all", moe_eval_capacity_token_fraction=1.0,
    )


class NllbExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self):
        return {"input_ids": torch.tensor([[4, 5, 6], [7, 8, 9]]),
                "decoder_input_ids": torch.tensor([[2, 10], [2, 11]])}

    def routing(self):
        module = upstream.NllbMoeSparseMLP(tiny_config(), ffn_dim=16).eval()
        return ExpertRouting(module, ContextVar("draft", default=False), layer_idx=3)

    def test_checkpoint_layout_and_target_logits_match_upstream(self):
        torch.manual_seed(19)
        original_forward = upstream.NllbMoeSparseMLP.forward
        target = upstream.NllbMoeForConditionalGeneration(tiny_config()).eval()
        extended = NllbMoeForConditionalGeneration(tiny_config()).eval()
        extended.load_state_dict(target.state_dict(), strict=True)
        self.assertEqual(list(extended.state_dict()), list(target.state_dict()))
        self.assertIs(upstream.NllbMoeSparseMLP.forward, original_forward)
        with torch.no_grad():
            torch.testing.assert_close(extended(**self.inputs()).logits, target(**self.inputs()).logits)

    def test_draft_mode_only_changes_decoder_and_is_reset(self):
        model = NllbMoeForConditionalGeneration(tiny_config()).eval()
        model.set_num_fixed_experts(2)
        encoder = model._encoder_routing[0].module.router
        decoder = model._decoder_routing[0].module.router
        with patch.object(encoder, "forward", wraps=encoder.forward) as enc, \
                patch.object(decoder, "forward", wraps=decoder.forward) as dec, torch.no_grad():
            model(**self.inputs(), fixed_gating=True)
            enc.assert_called_once()
            dec.assert_not_called()
            self.assertFalse(model._draft_mode.get())
            model(**self.inputs())
            dec.assert_called_once()

    def test_draft_mode_is_reset_after_a_failed_forward(self):
        model = NllbMoeForConditionalGeneration(tiny_config())
        with patch.object(upstream.NllbMoeForConditionalGeneration, "forward", side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                model(**self.inputs(), fixed_gating=True)
        self.assertFalse(model._draft_mode.get())

    def test_inactive_experts_are_never_called(self):
        routing = self.routing()
        visited = []
        for index, expert in enumerate(routing.module.experts.values()):
            expert.register_forward_pre_hook(lambda module, args, index=index: visited.append((index, len(args[0]))))
        probabilities = torch.tensor([[0.25, 0, 0.75, 0], [0, 0, 1.0, 0]])
        first = probabilities.eq(probabilities.max(-1, keepdim=True).values).long()
        with patch.object(routing.module.router, "forward", return_value=(first, probabilities)):
            output, _ = routing.module(torch.randn(1, 2, 8))
        self.assertEqual(visited, [(0, 1), (2, 2)])
        self.assertEqual(output.shape, (1, 2, 8))

    def test_verification_replaces_pinned_experts(self):
        routing = self.routing()
        routing.num_fixed_experts = 2
        routing.fixed_expert = torch.tensor([0, 1])
        for index, expert in enumerate(routing.module.experts.values()):
            for child in expert.children():
                child._hf_hook = SimpleNamespace(pin=index < 2, offload=index >= 2)
        routing._replace_pinned(torch.tensor([[0, 0, 0.4, 0.6], [0, 0, 0.2, 0.8]]))
        self.assertEqual(set(routing.fixed_expert.tolist()), {2, 3})
        for index, expert in enumerate(routing.module.experts.values()):
            for child in expert.children():
                self.assertEqual(child._hf_hook.pin, index >= 2)
                self.assertTrue(child._hf_hook.offload)

    def test_affinity_mapping_keeps_existing_collision_policy(self):
        routing = self.routing()
        routing.num_fixed_experts = 3
        routing.fixed_expert = torch.tensor([1, 2, 3])
        routing.affinity_expert_map = [[3, 2, 1]] * 4
        probabilities = torch.tensor([[0.4, 0.6, 0, 0]])
        with patch.object(routing.module.router, "forward", return_value=(None, probabilities)):
            weights = routing._draft_weights(torch.randn(1, 1, 8), None)
        torch.testing.assert_close(weights, torch.tensor([[0, 0, 0, 0.5]]))

    def test_missing_affinity_is_generated_and_reused(self):
        routing = self.routing()
        with torch.no_grad():
            for index, expert in enumerate(routing.module.experts.values()):
                expert.fc1.weight.fill_(index)
                expert.fc2.weight.zero_()
        with tempfile.TemporaryDirectory() as folder, patch("modelings.nllb.AFFINITY_DIR", Path(folder)):
            table = routing.get_affinity_table()
            self.assertAlmostEqual(table[0, 1].item(), (8 * 16) ** 0.5, places=5)
            self.assertTrue(torch.equal(table, table.T))
            self.assertTrue((Path(folder) / "affinity_nllb_3.pt").is_file())
            restored = self.routing()
            torch.testing.assert_close(restored.get_affinity_table(), table)


if __name__ == "__main__":
    unittest.main()
