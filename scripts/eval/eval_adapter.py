"""Paired valid_seen rollouts for frozen Receiver v1; no expert action access."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scripts.common import (HIDDEN_SIZE, N_VALID_SEEN, RECEIVER_MAX_NEW_TOKENS,
    EVAL_MAX_STEPS, sha256_file, write_json)
from scripts.eval.controls import make_mismatched_message, make_random_message, build_mismatched_mapping
from scripts.alfworld_utils import make_plain_tw_env, parse_action, process_ob, strip_intro_text
from scripts.model.runtime import (BASE, BASE_REVISION, load_receiver, load_rows, adapted,
    text_message, latent, render, build, nll)

GROUPS = ['matched', 'no_comm', 'mismatched', 'mismatched_control', 'zero', 'random', 'text']


def task_type(row):
    return Path(row['game_file'].rstrip('/')).parent.parent.name.split('-')[0]


def mapping(rows):
    strata = {}
    for r in rows:
        strata.setdefault(task_type(r), []).append({'episode_id':r['id'], 'game_file':r['game_file'],
            'L_i':len(r['plan']['communication_token_ids'])})
    return build_mismatched_mapping(strata, 0)


@torch.no_grad()
def messages_for(receiver, row, donor):
    z = adapted(receiver,row)
    controlled, info = make_mismatched_message(receiver.adapter, latent(donor), len(z),
        float(z.norm()), torch.float32, torch.device('cuda'))
    random_z, random_info = make_random_message(len(z),HIDDEN_SIZE,0,row['id'],float(z.norm()),
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
def rollout(receiver,tok,row,message,max_tokens=RECEIVER_MAX_NEW_TOKENS,max_steps=EVAL_MAX_STEPS):
    env = make_plain_tw_env(row['game_file'])
    rec = {'id':row['id'],'success':False,'n_steps':0,'outputs':[],'actions':[],
           'observations':[],'hits_end':[],'admissible':[],'reason':None}
    try:
        state = env.reset()
        obs = strip_intro_text(state['feedback'])
        assert ' '.join(obs.split()) == ' '.join(row['sender_task'].split())
        r = render(tok,row['messages'][:2],generation=True)
        b = build(receiver,r,message)
        past, logits = advance(receiver,b['input_embeds'])
        end_id = tok.convert_tokens_to_ids('<|im_end|>')
        stop_ids = {end_id,tok.eos_token_id}
        emb = receiver.base_model.get_input_embeddings()
        for step in range(max_steps):
            ids = []; hit = False
            for _ in range(max_tokens):
                nxt = int(logits.argmax())
                if nxt in stop_ids:
                    hit = True; break
                ids.append(nxt)
                past,logits = advance(receiver,emb(torch.tensor([[nxt]],device='cuda')),past)
            text = tok.decode(ids)
            rec['outputs'].append(text); rec['hits_end'].append(hit)
            try:
                action = parse_action(text)
            except ValueError:
                rec['reason'] = 'parse_error'; break
            if not action:
                rec['reason'] = 'parse_error'; break
            rec['admissible'].append(action in state.get('admissible_commands',[]))
            rec['actions'].append(action)
            state,_,done = env.step(action)
            rec['n_steps'] += 1
            rec['observations'].append(process_ob(state['feedback']))
            if state['won']:
                rec['success'] = True; rec['reason'] = 'won'; break
            if done:
                rec['reason'] = 'done_not_won'; break
            # Generated IDs are already cached. Append only the fixed turn suffix.
            suffix = '<|im_end|>\n<|im_start|>user\nObservation: '+process_ob(state['feedback'])+'<|im_end|>\n<|im_start|>assistant\n'
            ids_suffix = tok.encode(suffix,add_special_tokens=False)
            past,logits = advance(receiver,emb(torch.tensor([ids_suffix],device='cuda')),past)
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
        'mean_steps':actions/len(records),'generated_turns':len(outputs),
        'single_action_turns':sum(s.count('Action:')==1 for s in outputs),
        'thought_action_turns':sum('Thought:' in s and s.count('Action:')==1 for s in outputs),
        'admissible_actions':sum(sum(r['admissible']) for r in records),
        'executed_actions':actions,'capped_turns':sum(sum(not h for h in r['hits_end']) for r in records),
        'errors':sum(r['reason']=='execution_error' for r in records)}


def paired(matched,control):
    by_id = {r['id']:r for r in control}
    d = np.array([int(r['success'])-int(by_id[r['id']]['success']) for r in matched])
    rng = np.random.default_rng(0)
    means = d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1)
    return {'difference':float(d.mean()), 'paired_bootstrap_95':np.quantile(means,[.025,.975]).tolist(),
        'matched_only':int((d==1).sum()),'control_only':int((d==-1).sum()),
        'warning':'One training seed; percentile interval can degenerate with no discordant pairs.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True);ap.add_argument('--latents',required=True)
    ap.add_argument('--train-dir',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--selection');ap.add_argument('--train-latents')
    a = ap.parse_args();out = Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    provenance = {'data':sha256_file(a.data),
        'latents_config':sha256_file(str(Path(a.latents)/'config.json')),
        'code':sha256_file(__file__),
        'checkpoints':{s:sha256_file(str(Path(a.train_dir)/(s+'.pt'))) for s in ('initial','best','last')},
        'selection':sha256_file(a.selection) if a.selection else None}
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==provenance,'evaluation resume provenance mismatch'
    else:
        write_json(str(out/'config.json'),provenance)
    receiver,tok,freeze = load_receiver()
    rows = load_rows(a.data,tok,a.latents)
    assert len(rows)==N_VALID_SEEN and all(r['split']=='valid_seen' for r in rows)
    donors = mapping(rows); write_json(str(out/'mismatched_mapping.json'),donors)
    lookup = {r['id']:r for r in rows}
    start = time.time(); all_records = {}; checkpoints = {}
    for stage in ('initial','best','last'):
        path = Path(a.train_dir)/(stage+'.pt')
        ckpt = torch.load(path,map_location='cpu',weights_only=False)
        assert ckpt['base']==BASE and ckpt['revision']==BASE_REVISION
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
                msgs,info = messages_for(receiver,row,donor)
                t0 = time.time()
                try:
                    rec = rollout(receiver,tok,row,msgs[group])
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
        'paired':{g:paired(all_records['best_matched'],all_records['best_'+g]) for g in GROUPS if g!='matched'},
        'data_sha256':sha256_file(a.data),'scope':'One seed, 140 valid_seen; no valid_unseen; no tuning on these outcomes.'}
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
                msgs,info = messages_for(receiver,row,sel[(i+1)%len(sel)])
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
