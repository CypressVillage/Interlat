"""CPU regression checks for the actual rollout loop; no model downloads."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from step1 import eval_rollout as ev


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 255

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = ''.join(m['role'] + ':' + m['content'] + '|' for m in messages)
        if add_generation_prompt:
            text += 'assistant:'
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


class Model:
    def get_input_embeddings(self):
        return torch.nn.Embedding(256, 2)

    def __call__(self, inputs_embeds, past_key_values=None, use_cache=True):
        return SimpleNamespace(past_key_values='updated', logits=torch.tensor([[[1., 2., 3.]]]))


class CacheTests(unittest.TestCase):
    def test_append_refreshes_logits(self):
        cache = {'past': None, 'logits': torch.zeros(1, 3)}
        ev.append_tokens(SimpleNamespace(base_model=Model()), cache, [1], 'cpu')
        self.assertEqual(cache['logits'].tolist(), [[1., 2., 3.]])

    def test_multiturn_history_and_terminal(self):
        for terminal_at, won, expected_reason in [(3, True, None), (1, False, 'done_not_won')]:
            with self.subTest(terminal_at=terminal_at):
                self.run_rollout(terminal_at, won, expected_reason)

    def run_rollout(self, terminal_at, won, expected_reason):
        tok = Tokenizer()
        count = {'steps': 0, 'generations': 0}
        env = SimpleNamespace(reset=lambda: {'feedback': 'initial'}, close=lambda: None)

        def step(action):
            count['steps'] += 1
            done = count['steps'] == terminal_at
            return {'feedback': 'next', 'won': done and won}, 0, done

        env.step = step

        def generate(receiver, cache, *args):
            count['generations'] += 1
            return list(f"Thought: inspect {count['generations']}\nAction: look".encode()), True

        def render(tok, messages):
            ids = tok.apply_chat_template(messages)
            return SimpleNamespace(input_ids=ids, injection_index=len(ids))

        with patch.object(ev, 'make_plain_tw_env', return_value=env), \
             patch.object(ev, 'bop_eop_ids', return_value=(0, 1)), \
             patch.object(ev, 'strip_intro_text', side_effect=lambda x: x), \
             patch.object(ev, 'split_task_and_initial_observation', return_value=('T', 'O')), \
             patch.object(ev, 'process_ob', side_effect=lambda x: x), \
             patch.object(ev, 'render_with_labels', side_effect=render), \
             patch.object(ev, 'greedy_generate', side_effect=generate):
            result = ev.rollout_episode(SimpleNamespace(base_model=Model()), tok,
                                        'unused', 'no_comm', None, 'cpu', max_steps=4)
        self.assertEqual(result['reason'], expected_reason)
        self.assertEqual(result['success'], won)
        self.assertEqual(count['generations'], terminal_at)


if __name__ == '__main__':
    unittest.main()
