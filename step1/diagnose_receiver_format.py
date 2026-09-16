"""Exploratory train-input-only check of the registered upstream prompt."""
import json
import torch
from step1.serialization import load_locked_tokenizer, receiver_initial_messages
from step1.adapter import load_frozen_receiver
from step1.eval_rollout import prefill, greedy_generate
from step1.env_utils import parse_action
from step1.common import write_json

ROOT = '/home/zhaobc/step1_smoke'
rows = [json.loads(line) for line in open(ROOT + '/manifests/subset-train4.jsonl')][:2]
inputs = json.load(open(ROOT + '/episodes/episode_inputs.json'))
tok = load_locked_tokenizer()
receiver, _ = load_frozen_receiver(tok, device='cuda')
receiver.eval()
results = []
for row in rows:
    info = inputs[row['episode_id']]
    messages = receiver_initial_messages(info['task_description'], info['initial_observation'])
    for variant, variant_messages in [('registered_upstream_prompt', messages)]:
        ids = tok.apply_chat_template(variant_messages,
                                     tokenize=True, add_generation_prompt=True)
        with torch.no_grad():
            embeds = receiver.base_model.get_input_embeddings()(torch.tensor([ids], device='cuda'))
            cache = prefill(receiver, embeds, 'cuda')
            tokens, hit = greedy_generate(receiver, cache, 100, tok.eos_token_id, 'cuda')
        output = tok.decode(tokens)
        try:
            action = parse_action(output)
        except ValueError:
            action = None
        record = {'episode_id': row['episode_id'], 'source_split': row['source_split'],
                  'variant': variant, 'messages': variant_messages,
                  'input_ids': ids, 'output': output, 'hit_terminator': hit,
                  'action': action, 'scope': 'exploratory format only, no environment rollout'}
        results.append(record)
        print(json.dumps(record), flush=True)
write_json(ROOT + '/diagnosis_20260909/format_diagnostic.json', results)
