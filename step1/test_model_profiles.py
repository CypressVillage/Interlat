"""Model-profile and width-sensitive control tests; no model downloads."""

import unittest

import torch

from step1.adapter import Step1Adapter, _supervised_token_ce
from step1.common import (
    DEFAULT_MODEL_PROFILE,
    EXPECTED_PARAM_COUNT,
    LATENT_EXPECTED_DIM,
    MODEL_ID,
    MODEL_REVISION,
    get_model_profile,
)
from step1.controls import make_random_message, make_zero_message


class ModelProfileTests(unittest.TestCase):
    def test_legacy_aliases_still_select_default_profile(self):
        profile = get_model_profile(DEFAULT_MODEL_PROFILE)
        self.assertEqual(MODEL_ID, profile["model_id"])
        self.assertEqual(MODEL_REVISION, profile["model_revision"])
        self.assertEqual(LATENT_EXPECTED_DIM, profile["hidden_size"])
        self.assertEqual(EXPECTED_PARAM_COUNT, profile["expected_param_count"])

    def test_adapter_parameter_counts_match_profiles(self):
        for name in ("qwen2.5-0.5b", "qwen2.5-7b"):
            profile = get_model_profile(name)
            with torch.device("meta"):
                adapter = Step1Adapter(
                    hidden_size=profile["hidden_size"],
                    num_heads=profile["adapter_num_heads"],
                )
            self.assertEqual(adapter.param_count(), profile["expected_param_count"])

    def test_controls_use_requested_hidden_size(self):
        hidden_size = 32
        zero = make_zero_message(3, hidden_size, torch.float32, torch.device("cpu"))
        random, _ = make_random_message(
            3, hidden_size, 2101, "episode", 2.0, torch.float32, torch.device("cpu")
        )
        self.assertEqual(tuple(zero.shape), (3, hidden_size))
        self.assertEqual(tuple(random.shape), (3, hidden_size))
        self.assertAlmostEqual(float(random.norm()), 2.0, places=5)

    def test_compact_supervised_ce_matches_full_logits(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 5, 4)
        labels = torch.tensor([
            [-100, -100, 2, 1, -100],
            [-100, 3, -100, 0, 2],
        ])
        head = torch.nn.Linear(4, 5, bias=False)
        full_logits = head(hidden)[:, :-1, :]
        expected = torch.nn.functional.cross_entropy(
            full_logits.reshape(-1, 5), labels[:, 1:].reshape(-1), ignore_index=-100
        )
        actual, compact_logits, n_tokens = _supervised_token_ce(head, hidden, labels)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(compact_logits.shape, (5, 5))
        self.assertEqual(n_tokens, 5)


if __name__ == "__main__":
    unittest.main()
