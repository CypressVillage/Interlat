import unittest

from step1.env_utils import parse_action


class ActionCleanupTests(unittest.TestCase):
    def test_trailing_fence(self):
        self.assertEqual(parse_action('Thought: search\nAction: go to desk 1\n``` '), 'go to desk 1')

    def test_boundary_markers_only(self):
        self.assertEqual(parse_action('Action: `take book 1 from desk 1`'), 'take book 1 from desk 1')
        self.assertEqual(parse_action('Action: put book 1 in/on desk 1'), 'put book 1 in/on desk 1')

    def test_missing_action(self):
        with self.assertRaises(ValueError):
            parse_action('no action here')
