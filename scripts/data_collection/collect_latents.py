"""Offline paper-prompt collection; upstream sampling defaults, aligned full states.

Uses batched incremental decoding without retaining all-layer generation outputs.
Each task has an independent RNG, allowing resumable collection. No quality filter.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.common import (SENDER_MODEL as MODEL, SENDER_REVISION as REVISION,
    SENDER_MAX_NEW_TOKENS, PREPARED_DIR, DEFAULT_RUN_DIR, sha256_file, write_json)


def sender_prompt(task):
    text = '\n    Please provide a general plan to solve this task.\n\n\n    The task is: '+task+'\n    '
    return '<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n'


def sample(logits, generator):
    scores = logits.float() / .7
    values, indices = scores.topk(50)
    probs = values.softmax(-1)
    mask = probs.cumsum(-1) - probs > .9
    probs = probs.masked_fill(mask, 0)
    return indices[torch.multinomial(probs, 1, generator=generator)].item()


@torch.no_grad()
def collect_batch(model, tok, rows, cap):
    inputs = tok([sender_prompt(r['sender_task']) for r in rows], padding=True,
                 return_tensors='pt').to('cuda')
    ids, mask = inputs.input_ids, inputs.attention_mask
    rngs = [torch.Generator(device='cuda').manual_seed(
        int(hashlib.sha256(('42:'+r['id']).encode()).hexdigest()[:15], 16)) for r in rows]
    tokens = [[] for r in rows]
    states = [[] for r in rows]
    done = [False] * len(rows)
    past = None
    for step in range(cap):
        pos = mask.long().cumsum(-1)-1
        pos.masked_fill_(mask == 0, 0)
        if past is not None:
            pos = pos[:, -1:]
        out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                          past_key_values=past, use_cache=True, return_dict=True)
        h = out.last_hidden_state[:, -1]
        logits = model.lm_head(h)
        next_ids = []
        for i in range(len(rows)):
            t = tok.eos_token_id if done[i] else sample(logits[i], rngs[i])
            next_ids.append(t)
            if not done[i]:
                tokens[i].append(t)
                states[i].append(h[i].cpu())
                done[i] = t == tok.eos_token_id
        if all(done):
            break
        past = out.past_key_values
        ids = torch.tensor(next_ids, device='cuda')[:, None]
        mask = torch.cat((mask, torch.ones_like(ids)), dim=1)
    return [(ts, torch.stack(hs)) for ts, hs in zip(tokens, states)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data', nargs='+', default=[f'{PREPARED_DIR}/fit.jsonl', f'{PREPARED_DIR}/selection.jsonl'], help='prepared task JSONL files')
    ap.add_argument('--out-dir', default=f'{DEFAULT_RUN_DIR}/latents', help='Sender cache directory')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-new-tokens', type=int, default=SENDER_MAX_NEW_TOKENS)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for p in a.data for line in Path(p).read_text().splitlines()]
    if a.limit:
        rows = rows[:a.limit]
    assert len({r['id'] for r in rows}) == len(rows)
    config = {'model': MODEL, 'revision': REVISION, 'temperature': .7, 'top_p': .9,
              'top_k': 50, 'repetition_penalty': 1., 'num_beams': 1,
              'max_new_tokens': a.max_new_tokens, 'batch_size': a.batch_size,
              'rng': 'task-specific SHA256(42:id) first 15 hex digits',
              'attention': 'sdpa', 'dtype': 'bfloat16',
              'data': {p: sha256_file(p) for p in a.data},
              'latent': 'last layer immediately BEFORE predicting each token; includes EOS; no truncation',
              'limit': a.limit}
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text()) == config, 'resume config differs'
    else:
        write_json(str(out/'config.json'), config)
    pending = []
    for r in rows:
        p = out/(r['id']+'.json')
        if p.exists():
            meta = json.loads(p.read_text())
            assert sha256_file(str(out/(r['id']+'.pt'))) == meta['sha256']
        else:
            pending.append(r)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side='left')
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    start = time.time()
    for j in range(0, len(pending), a.batch_size):
        chunk = pending[j:j+a.batch_size]
        results = collect_batch(model, tok, chunk, a.max_new_tokens)
        for r, (ids, h) in zip(chunk, results):
            tensor_path = out/(r['id']+'.pt')
            torch.save(h, tensor_path)
            meta = {'id': r['id'], 'token_ids': ids, 'text': tok.decode(ids, skip_special_tokens=True),
                    'ended_eos': ids[-1] == tok.eos_token_id, 'shape': list(h.shape),
                    'sha256': sha256_file(str(tensor_path)), 'split': r['split'],
                    'prompt': sender_prompt(r['sender_task'])}
            write_json(str(out/(r['id']+'.json')), meta)
        print(json.dumps({'completed': j+len(chunk), 'pending_at_start': len(pending),
              'elapsed_s': time.time()-start, 'lengths': [len(x[0]) for x in results],
              'ids': [r['id'] for r in chunk]}), flush=True)
    metas = [json.loads((out/(r['id']+'.json')).read_text()) for r in rows]
    write_json(str(out/'summary.json'), {'count': len(metas),
        'tokens': sum(len(m['token_ids']) for m in metas),
        'capped': sum(not m['ended_eos'] for m in metas), 'elapsed_this_run_s': time.time()-start})


if __name__ == '__main__':
    main()
