"""Completion audit and report for a current frozen-Receiver run, including continuation.

This is read-only with respect to experiment inputs/checkpoints. It refuses to
claim completion from a pipeline launch or incomplete episode files.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from step1.common import (RECEIVER_MODEL, ADAPTER_PARAM_COUNT, GLOBAL_BATCH,
    TRAIN_SEED, N_VALID_SEEN, sha256_file, write_json)

GROUPS = ('matched','no_comm','mismatched','mismatched_control','zero','random','text')


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def wilson(successes,n,z=1.959963984540054):
    p=successes/n
    den=1+z*z/n
    middle=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,middle-half),min(1.,middle+half)]


def comparison(matched,control):
    mapping={r['id']:r for r in control}
    d=np.array([int(r['success'])-int(mapping[r['id']]['success']) for r in matched])
    wins=int((d==1).sum());losses=int((d==-1).sum());discordant=wins+losses
    p=1. if not discordant else min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant)
    rng=np.random.default_rng(0)
    boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
    return {'delta':float(d.mean()),'matched_only':wins,'control_only':losses,
        'paired_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
        'mcnemar_exact_p_unadjusted':p,'bootstrap_degenerate':bool(not discordant)}


def summarize(records):
    n=len(records);successes=sum(r['success'] for r in records)
    turns=sum(len(r['outputs']) for r in records)
    actions=sum(len(r['actions']) for r in records)
    return {'n':n,'successes':successes,'success_rate':successes/n,'success_wilson_95':wilson(successes,n),
        'parse_error_episodes':sum(r['reason']=='parse_error' for r in records),
        'errors':sum(r['reason']=='execution_error' for r in records),
        'mean_steps':sum(r['n_steps'] for r in records)/n,
        'generated_turns':turns,'actions':actions,
        'single_action_turns':sum(s.count('Action:')==1 for r in records for s in r['outputs']),
        'thought_action_turns':sum('Thought:' in s and s.count('Action:')==1 for r in records for s in r['outputs']),
        'admissible_actions':sum(sum(r['admissible']) for r in records),
        'capped_turns':sum(sum(not h for h in r['hits_end']) for r in records),
        'wall_seconds_sum':sum(r['seconds'] for r in records)}


def training_records(directory, visited=None):
    """Follow verified continuation provenance without counting later parent updates."""
    directory = directory.resolve()
    visited = set() if visited is None else visited
    if directory in visited:
        raise ValueError('cyclic training continuation provenance')
    visited.add(directory)
    config = load(directory/'config.json')
    records = lines(directory/'training.jsonl')
    resume = config.get('resume')
    if not resume:
        return records
    checkpoint = Path(resume['source_checkpoint'])
    if sha256_file(checkpoint) != resume['checkpoint_sha256']:
        raise ValueError('continuation source checkpoint changed')
    parent = checkpoint.parent
    if sha256_file(parent/'training.jsonl') != resume['source_training_log_sha256']:
        raise ValueError('continuation source training log changed')
    previous = training_records(parent, visited)
    prefix = [r for r in previous if r.get('step', 0) <= resume['step']]
    return prefix + records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,help='artifact root containing cache audits and valid_seen.jsonl')
    ap.add_argument('--train-dir',help='default: <root>/train; may point to a continuation directory')
    ap.add_argument('--eval-dir',help='default: <root>/eval')
    a=ap.parse_args();root=Path(a.root)
    train=Path(a.train_dir) if a.train_dir else root/'train'
    evaluation=Path(a.eval_dir) if a.eval_dir else root/'eval'
    # Required terminal evidence. Missing files intentionally raise rather than
    # turning an active or failed job into a completed negative experiment.
    complete=load(evaluation/'complete.json')
    cfg=load(train/'config.json');training=load(train/'summary.json')
    assert training['updates']>0 and training['base_unchanged']
    assert load(train/'base_before.json')['sha256']==training['base_sha256']
    assert cfg['freeze']['base_trainable']==cfg['freeze']['embed_trainable']==cfg['freeze']['lm_head_trainable']==0
    assert cfg['freeze']['trainable_total']==ADAPTER_PARAM_COUNT
    assert cfg['base']==RECEIVER_MODEL and cfg['seed']==TRAIN_SEED
    assert cfg['global_batch']==GLOBAL_BATCH
    planned=math.ceil(cfg['fit_count']/cfg['global_batch'])*cfg['epochs']
    assert cfg['max_updates']==planned and training['updates']<=planned
    if training['stop_reason']=='epoch_budget':assert training['updates']==planned
    else:assert training['stop_reason']=='wall_budget'
    for name,digest in training['checkpoint_sha256'].items():
        assert sha256_file(str(train/name))==digest,name
    for name,digest in complete['artifacts_sha256'].items():
        assert sha256_file(str(evaluation/name))==digest,name
    train_audit=load(root/'cache_audit.json');eval_audit=load(root/'eval_cache_audit.json')
    collection={split:load(root/folder/'summary.json') for split,folder in
        (('fit_and_selection','latents'),('valid_seen','eval_latents'))}
    total_train=cfg['fit_count']+cfg['selection_count']
    assert collection['fit_and_selection']['count']==total_train and collection['valid_seen']['count']==N_VALID_SEEN
    assert train_audit['count']==total_train and eval_audit['count']==N_VALID_SEEN
    assert train_audit['all_payload_checksums_verified'] and train_audit['all_states_finite']
    assert eval_audit['all_payload_checksums_verified'] and eval_audit['all_states_finite']
    for name in ('train_cache_exact.json','eval_cache_exact.json'):
        exact=load(root/name)
        assert exact['passed']
        assert all(r['positions']==r['bitwise_equal_positions'] for r in exact['records'])
    fit=lines(Path(cfg['fit']));selection=lines(Path(cfg['selection']));seen=lines(root/'valid_seen.jsonl')
    fit_ids={r['id'] for r in fit};selection_ids={r['id'] for r in selection};seen_ids={r['id'] for r in seen}
    assert len(fit_ids)==cfg['fit_count'] and len(selection_ids)==cfg['selection_count'] and len(seen_ids)==N_VALID_SEEN
    assert fit_ids.isdisjoint(selection_ids) and fit_ids.isdisjoint(seen_ids) and selection_ids.isdisjoint(seen_ids)
    assert all('/valid_seen/' in r['game_file'] for r in seen)
    update_log=training_records(train);updates=[r for r in update_log if r['event']=='update']
    assert len(updates)==training['updates']
    encountered=set()
    for i,r in enumerate(updates):
        assert r['step']==i+1 and 1<len(r['ids'])<=cfg['global_batch']
        assert set(r['ids'])<=fit_ids and set(r['donors'])==set(r['ids'])
        assert all(x!=y for x,y in zip(r['ids'],r['donors']))
        assert all(x in {n/10 for n in range(11)} for x in r['fractions'])
        assert np.isfinite(r['losses']).all() and math.isfinite(r['grad_norm_before_clip'])
        assert -math.log(2)-1e-5<=r['losses'][1]<=1e-5
        assert r['losses'][2]>=-1e-5
        encountered.update(r['ids'])
    assert sum(len(r['ids']) for r in updates)==training['examples']
    vals=[{'step':0,'nll':training['initial_nll']}]+[
        {'step':r['step'],'nll':r['nll']} for r in update_log if r['event']=='validation']
    assert min(vals,key=lambda r:r['nll'])==training['best']
    record_sets={}
    for stage in ('initial','best','last'):
        for group in GROUPS if stage=='best' else ('matched',):
            key=stage+'_'+group;records=lines(evaluation/(key+'.jsonl'))
            assert len(records)==N_VALID_SEEN and {r['id'] for r in records}==seen_ids,key
            assert all(r['stage']==stage and r['group']==group for r in records)
            assert all(0<=r['n_steps']<=20 for r in records)
            assert all(len(r['actions'])==r['n_steps'] for r in records)
            for r in records:
                for name in ('controlled','random'):
                    norm=r['message_info'][name]
                    assert math.isclose(norm['actual_norm_float64'],norm['target_norm_float64'],rel_tol=1e-6)
                    assert abs(norm['injected_bf16_norm_ratio']-1)<1e-3
            record_sets[key]=records
    summary={k:summarize(v) for k,v in record_sets.items()}
    pairs={g:comparison(record_sets['best_matched'],record_sets['best_'+g]) for g in GROUPS if g!='matched'}
    teacher=load(evaluation/'selection_summary.json')
    for group in GROUPS:
        records=lines(evaluation/('selection_'+group+'.jsonl'))
        assert len(records)==cfg['selection_count'] and {r['id'] for r in records}==selection_ids
        expected=sum(r['nll']*r['tokens'] for r in records)/sum(r['tokens'] for r in records)
        assert math.isclose(expected,teacher[group]['nll'],rel_tol=1e-10)
    errors=sum(v['errors'] for v in summary.values())
    facts={'completion_verified':True,'behavioral_comparisons_valid':errors==0,
        'execution_errors':errors,'training':training,'fit_unique_exposed':len(encountered),
        'fit_unique_unexposed':len(fit_ids-encountered),
        'fit_replay_failures':sum(not r['current_environment_audit']['success'] for r in fit),
        'training_token_presentations':sum(r['tokens'] for r in updates),
        'fit_presentation_equivalent_epochs':training['examples']/len(fit),
        'training_update_wall_seconds':sum(r['update_s'] for r in updates),
        'realized_schedule_ranges':{name:[min(r['weights'][name] for r in updates),
            max(r['weights'][name] for r in updates)] for name in ('lr','sep','align')},
        'peak_training_allocated_bytes':max(r['peak_allocated_bytes'] for r in updates),
        'max_gradient_norm_before_clip':max(r['grad_norm_before_clip'] for r in updates),
        'updates_with_gradient_clipping':sum(r['grad_norm_before_clip']>1 for r in updates),
        'selection_curve':vals,'groups':summary,'paired_success':pairs,'selection_nll':teacher,
        'sender_train_audit':train_audit,'sender_eval_audit':eval_audit,
        'sender_collection':collection,
        'scope':'Frozen Base, one training seed and one control draw, valid_seen only; not equivalent paper replication',
        'source_hashes':cfg['source_hashes'],
        'training_directory':str(train.resolve()), 'evaluation_directory':str(evaluation.resolve()),
        'audit_script_sha256':sha256_file(__file__)}
    write_json(str(root/'completion_audit.json'),facts)
    text=['# Frozen Receiver v1: training and evaluation report','',
        '## Validity and scope','',
        f'All required terminal artifacts and episode counts were verified. Execution errors: **{errors}**.',
        'The Receiver was Qwen2.5-7B Base, frozen throughout. Only the 77,113,346-parameter full-length adapter was trained.',
        'This is the user-authorized frozen-Receiver variant, not the advisor-approved historical CE-only protocol or an equivalent reproduction of paper performance.',
        'Both agents received the goal and initial observation; these results do not demonstrate delivery of previously unknown environment facts.','',
        '## Executed budget','',
        f'- Offline Sender collection: {sum(r["count"] for r in collection.values())} tasks, {sum(r["tokens"] for r in collection.values()):,} generated tokens, {sum(r["capped"] for r in collection.values())} capped plans; recorded collection wall time {sum(r["elapsed_this_run_s"] for r in collection.values())/3600:.2f} h.',
        f'- Updates: {training["updates"]}/{planned}; stop: `{training["stop_reason"]}`.',
        f'- Fit presentations: {training["examples"]}; unique fit tasks exposed: {len(encountered)}/{len(fit_ids)}.',
        f'- Presentation-equivalent epochs: {facts["fit_presentation_equivalent_epochs"]:.3f}; an epoch is not an optimizer update.',
        f'- Total supervised-token presentations: {facts["training_token_presentations"]:,}.',
        f'- Update wall time across verified training lineage: {facts["training_update_wall_seconds"]/3600:.2f} h; current-session logged wall time: {training["elapsed_s"]/3600:.2f} h (excludes initial model/data loading).',
        f'- Realized lr / separation / alignment ranges: `{facts["realized_schedule_ranges"]}`. A wall-budget stop may precede the planned annealing endpoint.',
        f'- Updates using gradient clipping: {facts["updates_with_gradient_clipping"]}/{training["updates"]}; this alone does not diagnose instability.',
        f'- Peak training CUDA allocation: {facts["peak_training_allocated_bytes"]/2**30:.2f} GiB (not whole-machine memory).',
        f'- Selected update: {training["best"]["step"]}; initial/selected/last selection NLL: {training["initial_nll"]:.6f} / {training["best"]["nll"]:.6f} / {training["last_nll"]:.6f}.',
        f'- Full Base checksum before and after: `{training["base_sha256"]}`.',
        f'- One seed ({cfg["seed"]}). Selection used {len(selection_ids)} training-source holdouts; performance used the independent {len(seen_ids)} canonical valid_seen tasks.',
        '- All published trajectories were retained without action corrections, including known replay failures.','',
        '## Independent task behavior','',
        '| Stage / group | Successes / 140 | Success 95% Wilson interval | Parse-error episodes | Admissible / executed actions | Capped / generated turns |',
        '|---|---:|---:|---:|---:|---:|']
    for key,r in summary.items():
        lo,hi=r['success_wilson_95']
        text.append(f'| {key} | {r["successes"]} | {100*lo:.2f}–{100*hi:.2f}% | {r["parse_error_episodes"]} | {r["admissible_actions"]}/{r["actions"]} | {r["capped_turns"]}/{r["generated_turns"]} |')
    text+=['','## Paired matched-minus-control task success','',
        '| Control | Difference (percentage points) | Paired bootstrap 95% | Matched-only / control-only |',
        '|---|---:|---:|---:|']
    for g,p in pairs.items():
        lo,hi=p['paired_bootstrap_95']
        text.append(f'| {g} | {100*p["delta"]:.2f} | {100*lo:.2f}–{100*hi:.2f} | {p["matched_only"]}/{p["control_only"]} |')
    text+=['','A zero-width bootstrap interval with zero discordant pairs does **not** prove equivalence. Multiple control comparisons are exploratory; no across-seed claim is supported.','',
        '## Teacher-forced diagnostics (selection split)','',
        '| Group | Response-token NLL |','|---|---:|']
    for g in GROUPS:text.append(f'| {g} | {teacher[g]["nll"]:.6f} |')
    text+=['','Selection NLL is not an independent task-performance result and cannot by itself establish useful latent communication.','',
        '## Interpretation limits','',
        '- Useful behavior requires actual environment actions and task completion, not merely a lower SFT loss or parsable syntax.',
        '- Read matched-vs-control success together with formatting, admissibility, initial/last checkpoints, and the actual update budget.',
        '- No positive outcome is required for this experiment to be complete. A negative outcome is specific to this budget, model pair, data quality, and explicitly chosen schedule.',
        '- Public paper/code discrepancies, BF16 SDPA versus FlashAttention-2, native accumulation versus DeepSpeed, GB10/ARM environment differences, and the one-seed limit are recorded in `step1/docs/FROZEN_RECEIVER_V1.md`.',
        '- The failed approximate full-prefill alignment diagnostic remains saved; exact same-batch prediction-position replays passed.','',
        '## Traceability','',
        f'Artifacts: `{root}`. Training: `{train}`. Evaluation: `{evaluation}`. Machine-readable checks: `completion_audit.json`. Source hashes: training `config.json`; continuation checkpoint/log lineage is verified. No remote publication or Issue-state change was performed.','']
    (root/'report.md').write_text('\n'.join(text))
    print(json.dumps({'completion_verified':True,'behavioral_comparisons_valid':errors==0,
        'updates':training['updates'],'best_step':training['best']['step'],
        'best_matched_successes':summary['best_matched']['successes'],'report':str(root/'report.md')}),flush=True)


if __name__=='__main__':main()
