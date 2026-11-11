"""Paired valid_seen rollouts for frozen Receiver v1; no expert action access."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from configs.config import config_for_parser, load_config
from utils.hashing import sha256_file, write_json
from eval.controls import make_mismatched_message, make_random_message, build_mismatched_mapping
from alfworld_utils import make_plain_tw_env, process_ob, strip_intro_text
from eval.turns import PROTOCOL, action_turn, cache_length, crop_cache
from model.receiver import load_receiver
from model.runtime import (load_rows, adapted,
    text_message, latent, render, build, nll)

GROUPS = ['matched', 'no_comm', 'mismatched', 'mismatched_control', 'zero', 'random', 'text']


def task_type(row):
    return Path(row['game_file'].rstrip('/')).parent.parent.name.split('-')[0]


def mapping(rows, seed=None):
    seed = seed if seed is not None else load_config()['evaluation']['seed']
    strata = {}
    for r in rows:
        strata.setdefault(task_type(r), []).append({'episode_id':r['id'], 'game_file':r['game_file'],
            'L_i':len(r['plan']['communication_token_ids'])})
    return build_mismatched_mapping(strata, seed)


@torch.no_grad()
def messages_for(receiver, row, donor, seed=None):
    seed = seed if seed is not None else load_config()['evaluation']['seed']
    z = adapted(receiver,row)
    controlled, info = make_mismatched_message(receiver.adapter, latent(donor, receiver.adapter.hidden_size), len(z),
        float(z.norm()), torch.float32, torch.device('cuda'))
    random_z, random_info = make_random_message(len(z),receiver.adapter.hidden_size,seed,row['id'],float(z.norm()),
        torch.float32,torch.device('cuda'))
    # The legacy helper measures random norms on CPU in float32. At millions of
    # coordinates its reduction can differ from the device norm. Recalibrate both
    # controls against the same float64 reference before the common bf16 cast.
    target_norm = z.double().norm()
    for message, metadata in ((controlled,info),(random_z,random_info)):
        correction = target_norm/message.double().norm()
        message.mul_(correction.to(message.dtype))
        metadata['float64_norm_correction'] = float(correction)
        metadata['target_norm_float64'] = float(target_norm)
        metadata['actual_norm_float64'] = float(message.double().norm())
        metadata['injected_bf16_norm_ratio'] = float(message.bfloat16().double().norm()/z.bfloat16().double().norm())
    return {'matched':z, 'no_comm':None, 'mismatched':adapted(receiver,donor),
        'mismatched_control':controlled,'zero':torch.zeros_like(z),'random':random_z,
        'text':text_message(receiver,row)}, {'donor':donor['id'], 'controlled':info,'random':random_info}


@torch.no_grad()
def advance(receiver, embeds, past=None):
    out = receiver.base_model.model(inputs_embeds=embeds, past_key_values=past,
        use_cache=True, return_dict=True)
    return out.past_key_values, receiver.base_model.lm_head(out.last_hidden_state[:,-1])


@torch.no_grad()
def generate_turn(receiver, tok, past, logits, max_tokens):
    """Stop at the first complete action line, before imagined future turns."""
    stop_ids = {tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id}
    emb = receiver.base_model.get_input_embeddings()
    ids = []
    for _ in range(max_tokens):
        nxt = int(logits.argmax())
        if nxt in stop_ids:
            return tok.decode(ids), past, 'eos', len(ids)
        ids.append(nxt)
        text = tok.decode(ids)
        # Check BEFORE advancing: one token can contain both newline and the
        # beginning of a fabricated Thought/Observation. The caller rolls back
        # and replays only the accepted assistant prefix before the next turn.
        try:
            complete = action_turn(text) is not None
        except ValueError:
            complete = True  # An empty terminated action is a parse failure.
        if complete:
            return text, past, 'action_line', len(ids)
        past, logits = advance(receiver, emb(torch.tensor([[nxt]], device=emb.weight.device)), past)
    return tok.decode(ids), past, 'max_tokens', len(ids)


@torch.no_grad()
def rollout(receiver,tok,row,message,max_tokens=None,max_steps=None):
    if max_tokens is None or max_steps is None:
        config = load_config()['evaluation']
        max_tokens = max_tokens if max_tokens is not None else config['max_new_tokens']
        max_steps = max_steps if max_steps is not None else config['max_steps']
    env = make_plain_tw_env(row['game_file'])
    rec = {'id':row['id'],'success':False,'n_steps':0,'outputs':[],'actions':[],
           'observations':[],'hits_end':[],'admissible':[],'reason':None,
           'raw_outputs':[],'stop_reasons':[],'generated_tokens':[], 'protocol':PROTOCOL}
    try:
        state = env.reset()
        obs = strip_intro_text(state['feedback'])
        assert ' '.join(obs.split()) == ' '.join(row['sender_task'].split())
        r = render(tok,row['messages'][:2],generation=True)
        b = build(receiver,r,message)
        past, logits = advance(receiver,b['input_embeds'])
        emb = receiver.base_model.get_input_embeddings()
        for step in range(max_steps):
            turn_start = cache_length(past)
            raw, past, stop, token_count = generate_turn(receiver,tok,past,logits,max_tokens)
            rec['raw_outputs'].append(raw); rec['hits_end'].append(stop == 'eos')
            rec['stop_reasons'].append(stop); rec['generated_tokens'].append(token_count)
            try:
                parsed = action_turn(raw, final=stop == 'eos')
            except ValueError:
                rec['outputs'].append(raw)
                rec['reason'] = 'parse_error'; break
            if parsed is None:
                rec['outputs'].append(raw)
                rec['reason'] = 'max_tokens' if stop == 'max_tokens' else 'parse_error'; break
            text, action = parsed
            rec['outputs'].append(text)
            rec['admissible'].append(action in state.get('admissible_commands',[]))
            rec['actions'].append(action)
            state,_,done = env.step(action)
            rec['n_steps'] += 1
            rec['observations'].append(process_ob(state['feedback']))
            if state['won']:
                rec['success'] = True; rec['reason'] = 'won'; break
            if done:
                rec['reason'] = 'done_not_won'; break
            # Discard all speculative tokens from this turn, then commit only
            # the accepted text plus REAL environment feedback. Retokenization
            # also handles a token spanning the action newline and future text.
            past = crop_cache(past, turn_start)
            suffix = '<|im_end|>\n<|im_start|>user\nObservation: '+process_ob(state['feedback'])+'<|im_end|>\n<|im_start|>assistant\n'
            ids_suffix = tok.encode(text+suffix,add_special_tokens=False)
            past,logits = advance(receiver,emb(torch.tensor([ids_suffix],device=emb.weight.device)),past)
        else:
            rec['reason'] = 'max_steps'
        return rec
    finally:
        env.close()


def summarize(records):
    outputs = [s for r in records for s in r['outputs']]
    actions = sum(r['n_steps'] for r in records)
    return {'episodes':len(records),'successes':sum(r['success'] for r in records),
        'success_rate':sum(r['success'] for r in records)/len(records),
        'parse_failure_episodes':sum(r['reason']=='parse_error' for r in records),
        'token_limit_episodes':sum(r['reason']=='max_tokens' for r in records),
        'mean_steps':actions/len(records),'generated_turns':len(outputs),
        'single_action_turns':sum(s.count('Action:')==1 for s in outputs),
        'thought_action_turns':sum('Thought:' in s and s.count('Action:')==1 for s in outputs),
        'admissible_actions':sum(sum(r['admissible']) for r in records),
        'executed_actions':actions,
        'capped_turns':sum(r['stop_reasons'].count('max_tokens') for r in records),
        'action_line_stops':sum(r['stop_reasons'].count('action_line') for r in records),
        'errors':sum(r['reason']=='execution_error' for r in records)}


def paired(matched,control,config=None):
    config = config if config is not None else load_config()['evaluation']
    by_id = {r['id']:r for r in control}
    d = np.array([int(r['success'])-int(by_id[r['id']]['success']) for r in matched])
    rng = np.random.default_rng(config['seed'])
    means = d[rng.integers(0,len(d),size=(config['bootstrap_samples'],len(d)))].mean(1)
    return {'difference':float(d.mean()), 'paired_bootstrap_95':np.quantile(means,[.025,.975]).tolist(),
        'matched_only':int((d==1).sum()),'control_only':int((d==-1).sum()),
        'warning':'One training seed; percentile interval can degenerate with no discordant pairs.'}


def verify_output_provenance(out, provenance):
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text()) != provenance:
            raise ValueError('evaluation resume provenance mismatch; use a new output directory')
    else:
        if list(out.glob('*.jsonl')):
            raise ValueError('evaluation outputs exist without provenance; use a new output directory')
        write_json(str(out/'config.json'),provenance)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    experiment = config_for_parser(ap)
    evaluation = experiment['evaluation']
    RECEIVER_MODEL = experiment['model']['receiver']['name_or_path']
    RECEIVER_REVISION = experiment['model']['receiver']['revision']
    ap.add_argument('--data',default=experiment['data']['eval_tasks']['output_path'],help='prepared evaluation inputs')
    ap.add_argument('--latents',default=experiment['data']['collection']['eval_latents'],help='evaluation Sender cache')
    ap.add_argument('--train-dir',default=experiment['training']['output_dir'],help='training checkpoints')
    ap.add_argument('--out-dir',default=evaluation['output_dir'],help='evaluation output directory')
    ap.add_argument('--selection', nargs='?', const=str(Path(experiment['data']['split']['output_dir']) / 'selection.jsonl'), help='enable selection diagnostics; optional path overrides selection.jsonl in split.output_dir')
    ap.add_argument('--train-latents',default=experiment['data']['collection']['train_latents'],help='training Sender cache for optional selection diagnostics')
    a = ap.parse_args();out = Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    provenance = {'data':sha256_file(a.data), 'protocol':PROTOCOL,
        'runtime_code':{p:sha256_file(p) for p in (
            'eval/turns.py','alfworld_utils.py','model/runtime.py','model/receiver.py','eval/controls.py')},
        'experiment': {'settings': experiment, 'arguments': vars(a)},
        'latents_config':sha256_file(str(Path(a.latents)/'config.json')),
        'code':sha256_file(__file__),
        'checkpoints':{s:sha256_file(str(Path(a.train_dir)/(s+'.pt'))) for s in ('initial','best','last')},
        'selection':sha256_file(a.selection) if a.selection else None}
    verify_output_provenance(out, provenance)
    receiver,tok,freeze = load_receiver(config=experiment['model'])
    rows = load_rows(a.data,tok,a.latents)
    assert len(rows)==evaluation['num_tasks'] and all(r['split']=='valid_seen' for r in rows)
    donors = mapping(rows,evaluation['seed']); write_json(str(out/'mismatched_mapping.json'),donors)
    lookup = {r['id']:r for r in rows}
    start = time.time(); all_records = {}; checkpoints = {}
    for stage in ('initial','best','last'):
        path = Path(a.train_dir)/(stage+'.pt')
        ckpt = torch.load(path,map_location='cpu',weights_only=False)
        assert ckpt['base']==RECEIVER_MODEL and ckpt['revision']==RECEIVER_REVISION
        checkpoints[stage] = {'step':ckpt['step'],'sha256':sha256_file(str(path))}
        receiver.adapter.load_state_dict(ckpt['adapter']);receiver.eval()
        groups = GROUPS if stage=='best' else ['matched']
        for group in groups:
            key = stage+'_'+group; target = out/(key+'.jsonl')
            records = [json.loads(s) for s in target.read_text().splitlines()] if target.exists() else []
            done_ids = {r['id'] for r in records}
            assert len(done_ids)==len(records)
            for row in rows:
                if row['id'] in done_ids:
                    continue
                donor = lookup[donors['pairs'][row['id']]['donor_episode_id']]
                msgs,info = messages_for(receiver,row,donor,evaluation['seed'])
                t0 = time.time()
                try:
                    rec = rollout(receiver,tok,row,msgs[group],evaluation['max_new_tokens'],evaluation['max_steps'])
                except Exception as e:
                    rec = {'id':row['id'],'success':False,'n_steps':0,'outputs':[],
                        'actions':[],'admissible':[],'hits_end':[],'reason':'execution_error','error':repr(e)}
                rec.update({'stage':stage,'group':group,'message_info':info,'seconds':time.time()-t0})
                with target.open('a') as f:
                    f.write(json.dumps(rec)+'\n')
                records.append(rec)
                print(json.dumps({'stage':stage,'group':group,'completed':len(records),
                    'successes':sum(r['success'] for r in records),'elapsed_s':time.time()-start}),flush=True)
                del msgs
            all_records[key] = records
    summary = {'freeze':freeze,'checkpoints':checkpoints,'elapsed_s':time.time()-start,
        'groups':{k:summarize(v) for k,v in all_records.items()},
        'paired':{g:paired(all_records['best_matched'],all_records['best_'+g],evaluation) for g in GROUPS if g!='matched'},
        'data_sha256':sha256_file(a.data),'scope':'One seed, configured valid_seen set; no valid_unseen; no tuning on these outcomes.'}
    write_json(str(out/'rollout_summary.json'),summary)
    if a.selection:
        ckpt = torch.load(Path(a.train_dir)/'best.pt',map_location='cpu',weights_only=False)
        receiver.adapter.load_state_dict(ckpt['adapter']);receiver.eval()
        sel = load_rows(a.selection,tok,a.train_latents)
        # This is a teacher-forced diagnostic, not an independent performance test.
        # Global deterministic cycle is the raw paper mismatch; length/norm controls separately.
        report = {}
        for group in GROUPS:
            target = out/('selection_'+group+'.jsonl')
            records = [json.loads(s) for s in target.read_text().splitlines()] if target.exists() else []
            done_ids = {r['id'] for r in records}
            for i,row in enumerate(sel):
                if row['id'] in done_ids:
                    continue
                msgs,info = messages_for(receiver,row,sel[(i+1)%len(sel)],evaluation['seed'])
                with torch.no_grad():
                    r = {'id':row['id'],**nll(receiver,row,msgs[group]),'message_info':info}
                with target.open('a') as f:f.write(json.dumps(r)+'\n')
                records.append(r);del msgs
            report[group] = {'nll':sum(r['nll']*r['tokens'] for r in records)/sum(r['tokens'] for r in records),
                'tokens':sum(r['tokens'] for r in records),'count':len(records)}
            write_json(str(out/'selection_summary.json'),report)
    write_json(str(out/'complete.json'),{'elapsed_s':time.time()-start,
        'artifacts_sha256':{p.name:sha256_file(str(p)) for p in out.glob('*.jsonl')}})


if __name__=='__main__':main()
