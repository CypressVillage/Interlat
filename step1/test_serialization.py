"""Prompt-contract regression tests; no model downloads."""

import unittest

from step1.serialization import (
    ALFWORLD_ONE_SHOT_DEMO,
    ALFWORLD_RECEIVER_INSTRUCTION,
    ALFWORLD_SINGLE_ACTION_RULE,
    receiver_first_user_content,
    receiver_initial_messages,
    sender_user_content,
)


class SerializationTests(unittest.TestCase):
    def test_receiver_contains_fixed_one_shot_once(self):
        content = receiver_first_user_content("put the apple away", "You see a table 1.")

        self.assertEqual(content.count(ALFWORLD_RECEIVER_INSTRUCTION), 1)
        self.assertEqual(content.count(ALFWORLD_ONE_SHOT_DEMO), 1)
        self.assertEqual(content.count(ALFWORLD_SINGLE_ACTION_RULE), 1)
        self.assertIn("Action: go to countertop 1", content)
        self.assertIn("The task is: put the apple away", content)
        self.assertIn("Initial observation: You see a table 1.", content)

    def test_receiver_demo_is_in_first_user_turn(self):
        messages = receiver_initial_messages("task", "observation")

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn(ALFWORLD_ONE_SHOT_DEMO, messages[1]["content"])

    def test_sender_contains_full_action_interface_without_receiver_demo(self):
        content = sender_user_content("task", "observation")

        self.assertEqual(content.count(ALFWORLD_RECEIVER_INSTRUCTION), 1)
        self.assertNotIn(ALFWORLD_ONE_SHOT_DEMO, content)
        self.assertIn("Please provide a general plan", content)


if __name__ == "__main__":
    unittest.main()
