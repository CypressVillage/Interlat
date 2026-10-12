"""CPU regression checks for evaluation resume and preregistered power grid."""

import unittest

import numpy as np

from step1.eval_rollout import completed_episode_draws
from step1.power_check import ALPHA, critical_values, joint_probs, simulate_power


class EvaluationResumeTests(unittest.TestCase):
    def test_resume_preserves_remaining_draws(self):
        rows = [{"episode_id": "a"}, {"episode_id": "b"}]
        done = completed_episode_draws(
            [{"episode_id": "a", "draw": 1101}, {"episode_id": "b", "draw": 1101},
             {"episode_id": "a", "draw": 1102}], rows, [1101, 1102, 1103])
        self.assertNotIn(("b", 1102), done)
        self.assertNotIn(("a", 1103), done)
        self.assertEqual(len(done), 3)

    def test_duplicate_or_changed_draw_refused(self):
        rows = [{"episode_id": "a"}]
        with self.assertRaises(ValueError):
            completed_episode_draws([{"episode_id": "a", "draw": 1101}] * 2, rows, [1101])
        with self.assertRaises(ValueError):
            completed_episode_draws([{"episode_id": "a", "draw": 1102}], rows, [1101])


class PowerCheckTests(unittest.TestCase):
    def test_joint_probability_constraints_include_baseline(self):
        self.assertIsNone(joint_probs(.10, .40, .01))
        self.assertIsNone(joint_probs(.25, .10, .20))
        self.assertAlmostEqual(sum(joint_probs(.25, .20, .05)), 1.)

    def test_exact_critical_values(self):
        critical = critical_values(140, ALPHA)
        self.assertEqual(critical[0], 1)
        self.assertEqual(critical[1], 2)
        self.assertEqual(critical[10], 10)  # only 10/10 exceeds alpha=.00625

    def test_fixed_seed_simulation_reproducible(self):
        probs = joint_probs(.25, .20, .05)
        crit = critical_values(140, ALPHA)
        result = simulate_power(np.random.default_rng(20260820), probs, crit)
        self.assertEqual(result, simulate_power(np.random.default_rng(20260820), probs, crit))
        self.assertGreaterEqual(result, 0.)
        self.assertLessEqual(result, 1.)


if __name__ == "__main__":
    unittest.main()
