"""Frozen-Receiver v1 training. See README.md for current overrides."""
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from configs.config import config_for_parser, load_config
from utils.hashing import sha256_file, write_json
from model.receiver import load_receiver
from model.runtime import (load_rows, adapted, latent,
    text_message, supervised_hidden, nll, parameter_fingerprint)
from train.diagnostics import TrainingProbes


def communication_mix(text_embeddings, latent, latent_fraction):
    """Public-code curriculum: adapted latent prefix then text suffix.

    Enforce position-level correspondence rather than silently resampling.
    The runtime defines the content span (terminal Sender EOS excluded in v1).
    """
    if text_embeddings.shape != latent.shape or latent.ndim != 2:
        raise ValueError('text/latent must have equal [L,D] shapes and aligned token positions')
    if not 0 <= latent_fraction <= 1:
        raise ValueError('latent_fraction must be in [0,1]')
    k = int(round(latent_fraction * latent.shape[0]))
    return torch.cat((latent[:k], text_embeddings[k:]), dim=0)


def align_donor_latent(donor, target_length, noise_std=None):
    """Match upstream negative length handling before adapter self-attention.

    Longer/equal donors are sliced without noise. Shorter donors are repeated,
    sliced and perturbed with N(0, .01^2) noise at every resulting position.
    """
    if donor.ndim != 2 or donor.shape[0] == 0 or target_length < 1:
        raise ValueError('require nonempty [L,D] donor and positive target length')
    if donor.shape[0] >= target_length:
        return donor[:target_length]
    repeats = target_length // donor.shape[0] + 1
    repeated = donor.repeat(repeats, 1)[:target_length]
    if noise_std is None:
        noise_std = load_config()['training']['donor_noise_std']
    return repeated + torch.randn_like(repeated)*noise_std


def paper_losses(matched, mismatched, text, labels):
    """Use logits aligned by causally predicted token, not absolute sequence index.

    No implicit detach: the trainer explicitly controls each branch's gradients.
    """
    if matched.ndim != 2 or matched.shape != mismatched.shape or matched.shape != text.shape:
        raise ValueError('three branches need aligned [supervised_tokens,vocabulary] logits')
    if labels.shape != matched.shape[:1] or not labels.numel():
        raise ValueError('labels must contain a nonempty aligned token vector')
    if (labels < 0).any():
        raise ValueError('select supervised positions before computing objectives')

    lp, lq, lt = [
        F.log_softmax(x.float(), dim=-1)
        for x in (matched, mismatched, text)
    ]
    p, q = lp.exp(), lq.exp()
    lm = torch.logaddexp(lp, lq) - torch.log(torch.tensor(2.,device=lp.device)) # log(e^logp + e^logq) - log(2) = log((p+q)/2)
    js = .5 * (
        (p * (lp-lm)).sum(-1)
        + (q * (lq-lm)).sum(-1)
    )
    kl = (p * (lp-lt)).sum(-1).mean()
    cosine = (
        1 - F.cosine_similarity( matched.float(), text.float(), dim = -1 )
    ).mean()
    return {
        'task_ce': F.cross_entropy(matched.float(), labels),
        'negative_js': -js.mean(),
        'latent_to_text_kl': kl,
        'logit_cosine_distance': cosine
    }


def schedule(step, total, config=None):
    config = config if config is not None else load_config()['training']
    warmup = math.ceil(total*config['warmup_ratio'])
    lr_scale = (step+1)/warmup if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))
    return {'lr': config['learning_rate']*lr_scale}


def training_stop_reason(step, stop_after_updates, elapsed_s, max_hours):
    """Execution limits do not alter the full epoch-based learning-rate schedule."""
    if stop_after_updates > 0 and step >= stop_after_updates:
        return 'update_limit'
    if max_hours > 0 and elapsed_s >= max_hours*3600:
        return 'wall_budget'
    return None


def dynamic_loss_weights(negative_js, plan_similarity, config=None):
    """Upstream coefficient ranges/controller, retaining our existing objectives.

    Upstream expects a nonnegative contrast loss; use max(0, .69 - JS)
    only as its control signal, not as a replacement for negative-JS loss.
    Preserve differentiation through the coefficients as upstream does.
    """
    config = config if config is not None else load_config()['training']['loss_weights']
    contrast_signal = (config['js_reference'] + negative_js).clamp(min=0.)
    return {
        'sep': config['sep_min'] + config['sep_range'] * (contrast_signal / config['js_reference']).clamp(0., 1.),
        'align': config['align_min'] + config['align_range'] * (plan_similarity / config['plan_reference']).clamp(0., 1.),
    }


def epoch_batches(count, group_size, rng, fractions=None):
    fractions = fractions if fractions is not None else load_config()['training']['latent_fractions']
    order = list(range(count)); rng.shuffle(order)
    for offset in range(0, count, group_size):
        indices = order[offset:offset+group_size]
        assert len(indices) > 1, 'negative requires another task'
        shift = rng.randrange(1,len(indices))
        sampled = [rng.choice(fractions) for _ in indices]
        yield indices, shift, sampled


def loss_from_hidden(head, hm, hn, ht, labels, chunk_size=None, config=None):
    """Exact full-vocabulary objectives, checkpointed by supervised-token chunks."""
    config = config if config is not None else load_config()['training']
    chunk_size = chunk_size if chunk_size is not None else config['loss_chunk_size']
    def chunk(m, n, t, y):
        losses = paper_losses(head(m), head(n), head(t), y)
        return torch.stack([losses[k] for k in (
            'task_ce', 'negative_js', 'latent_to_text_kl', 'logit_cosine_distance')])
    components = hm.new_zeros(4, dtype=torch.float32)
    for s in range(0, len(labels), chunk_size):
        args = (hm[s:s+chunk_size], hn[s:s+chunk_size], ht[s:s+chunk_size], labels[s:s+chunk_size])
        part = checkpoint(chunk, *args, use_reentrant=False) if torch.is_grad_enabled() else chunk(*args)
        components = components+part*(len(args[-1])/len(labels))
    plan_similarity = components[2] + components[3]
    weights = dynamic_loss_weights(components[1], plan_similarity, config['loss_weights'])
    total = components[0] + weights['sep'] * components[1] + weights['align'] * plan_similarity
    return total, components, weights


@torch.no_grad()
def validate(receiver, rows, out):
    receiver.eval()
    records = []
    for row in rows:
        v = nll(receiver, row, adapted(receiver, row))
        records.append({'id': row['id'], **v})
    out.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return sum(r['nll']*r['tokens'] for r in records)/sum(r['tokens'] for r in records)


def resume_history(checkpoint_path, resume):
    """Find the historical best without substituting the resume weights for it.

    Require the original initial/best artifacts. Refuse a best from beyond the
    requested resume point rather than leak future checkpoint-selection results.
    This remains an explicitly optimizer-reset continuation, not exact replay.
    """
    source = Path(checkpoint_path).resolve().parent
    previous = resume['config']
    initial_path, best_path = source/'initial.pt', source/'best.pt'
    if not initial_path.is_file() or not best_path.is_file():
        raise ValueError('resume requires initial.pt and best.pt alongside the epoch checkpoint')
    summary_path = source/'summary.json'
    if summary_path.is_file():
        hashes = json.loads(summary_path.read_text()).get('checkpoint_sha256', {})
        for path in (Path(checkpoint_path).resolve(), initial_path, best_path):
            if path.name in hashes and sha256_file(path) != hashes[path.name]:
                raise ValueError(f'resume checkpoint checksum mismatch: {path}')

    candidates = []
    initial_nll = None
    for path in (initial_path, best_path, Path(checkpoint_path).resolve()):
        candidate = resume if path == Path(checkpoint_path).resolve() else torch.load(
            path, map_location='cpu', weights_only=False)
        if candidate['base'] != resume['base'] or candidate['revision'] != resume['revision']:
            raise ValueError(f'resume model mismatch: {path}')
        config = candidate['config']
        for key in ('fit_count', 'selection_count', 'global_batch', 'max_updates', 'source_hashes'):
            if config[key] != previous[key]:
                raise ValueError(f'resume history config mismatch ({key}): {path}')
        step, score = candidate['step'], candidate['selection_nll']
        if not isinstance(step, int) or not 0 <= step <= resume['step']:
            raise ValueError(f'historical checkpoint is beyond the resume point: {path}')
        if score is None or not math.isfinite(score):
            raise ValueError(f'resume history requires finite selection NLL: {path}')
        if path == initial_path:
            if step != 0:
                raise ValueError('initial.pt must be the step-zero checkpoint')
            initial_nll = score
        candidates.append({'path': str(path), 'step': step, 'nll': score})
        del candidate
    selected = min(candidates, key=lambda item: (item['nll'], item['step']))
    return {'initial_path': str(initial_path), 'initial_nll': initial_nll,
            'best_path': selected['path'],
            'best': {'step': selected['step'], 'nll': selected['nll']}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    experiment = config_for_parser(ap)
    training = experiment['training']
    TRAIN_SEED = training['seed']
    GLOBAL_BATCH = training['global_batch']
    RECEIVER_MODEL = experiment['model']['receiver']['name_or_path']
    RECEIVER_REVISION = experiment['model']['receiver']['revision']
    ap.add_argument('--fit', default=str(Path(experiment['data']['split']['output_dir']) / 'fit.jsonl'), help='prepared fit tasks')
    ap.add_argument('--selection', default=str(Path(experiment['data']['split']['output_dir']) / 'selection.jsonl'), help='prepared checkpoint-selection tasks')
    ap.add_argument('--latents', default=experiment['data']['collection']['train_latents'], help='Sender cache directory')
    ap.add_argument('--out-dir', default=training['output_dir'], help='new training output directory')

    ap.add_argument('--epochs', type=int, default=training['epochs'])
    ap.add_argument('--max-hours', type=float, default=training['max_hours'], help='optional wall-clock limit; 0 disables it')
    ap.add_argument('--stop-after-updates', type=int, default=training['stop_after_updates'],
                    help='stop at this global optimizer step without shortening the LR schedule; 0 disables it')
    ap.add_argument('--diagnostic-probes', help='JSON list of fit IDs and fixed latent_fraction values; measure at start and each update')
    ap.add_argument('--resume-epoch', help='completed-epoch checkpoint; restart AdamW from saved adapter weights')

    ap.add_argument('--smoke', action='store_true', help='two fit tasks, one update; fresh throwaway adapter')
    a = ap.parse_args()
    if a.max_hours < 0 or a.epochs < 1 or a.stop_after_updates < 0 or (a.smoke and (a.resume_epoch or a.stop_after_updates)):
        ap.error('require nonnegative limits, epochs >= 1, and no resume or stop-after-updates with smoke')
    out = Path(a.out_dir);
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)

    receiver, tok, freeze = load_receiver(checkpointing=True, config=experiment['model'])
    fit = load_rows(a.fit, tok, a.latents, limit=2 if a.smoke else 0)
    selection = fit if a.smoke else load_rows(a.selection, tok, a.latents)
    if a.smoke:
        fit = fit[:2]; selection = fit; a.epochs = 1
    assert not ({r['id'] for r in fit}&{r['id'] for r in selection}) or a.smoke

    group_size = GLOBAL_BATCH if not a.smoke else 2
    total_steps = math.ceil( len(fit) / group_size ) * a.epochs
    resume = None; history = None; start_epoch = 0; start_step = 0
    if a.resume_epoch:
        resume = torch.load(a.resume_epoch, map_location='cpu', weights_only=False)
        previous = resume['config']
        assert resume['base'] == RECEIVER_MODEL and resume['revision'] == RECEIVER_REVISION
        assert previous['fit_count'] == len(fit) and previous['selection_count'] == len(selection)
        assert previous['global_batch'] == group_size and previous['max_updates'] == total_steps
        for source, digest in previous['source_hashes'].items():
            if source != 'train/train_step1.py':
                assert sha256_file(source) == digest, f'resume source mismatch: {source}'
        start_step = resume['step']
        steps_per_epoch = math.ceil(len(fit)/group_size)
        assert start_step > 0 and start_step % steps_per_epoch == 0, 'require a completed epoch'
        start_epoch = start_step // steps_per_epoch
        assert start_epoch < a.epochs and resume['selection_nll'] is not None
        history = resume_history(a.resume_epoch, resume)
        receiver.adapter.load_state_dict(resume['adapter'])
    if a.stop_after_updates and a.stop_after_updates <= start_step:
        ap.error('stop-after-updates must be beyond the resume step')
    probes = TrainingProbes(receiver, fit, json.loads(Path(a.diagnostic_probes).read_text())) if a.diagnostic_probes else None
    config = {**vars(a), 'base': RECEIVER_MODEL, 'base_revision': RECEIVER_REVISION, 'seed': TRAIN_SEED,
        'fit_count': len(fit), 'selection_count': len(selection), 'global_batch': group_size,
        'serial_microbatch': 1, 'max_updates': total_steps, 'optimizer': 'AdamW',
        'experiment': {'settings': experiment, 'arguments': vars(a)},
        'weight_decay': training['weight_decay'], 'betas': training['betas'], 'eps': training['eps'], 'grad_clip': training['grad_clip'],
        'warmup_updates': math.ceil(total_steps*training['warmup_ratio']), 'lr_schedule': 'linear warmup then cosine',
        'lambda_sep': 'sep_min + sep_range * clip(max(0, js_reference - JS) / js_reference, 0, 1)',
        'lambda_align': 'align_min + align_range * clip((KL + cosine_distance) / plan_reference, 0, 1)',
        'loss_weight_policy': 'per-example full-supervised-token losses; differentiable coefficients; upstream controller only, original objectives retained',
        'alpha': 1., 'beta': 1.,
        'mix': 'uniform choice from configured latent_fractions; matched/donor latent prefix then current-task text suffix; round(r*L); same r for both branches',
        'training_mismatch': 'different-ID cyclic donor; raw latent truncated or repeated to current-task content length before adapter; repeated donors receive configured Gaussian noise; current-task text retained',
        'terminal_eos': 'retained in source cache; excluded from both text and latent communication',
        'gradient_policy': 'both mixed matched and mixed donor; text frozen/no_grad',
        'normalization': 'supervised-token mean within each optimizer batch, including partial last batch',
        'diagnostic_probe_sha256': sha256_file(a.diagnostic_probes) if a.diagnostic_probes else None,
        'diagnostic_probe_reference_step': start_step if probes else None,
        'source_hashes': {p: sha256_file(p) for p in [a.fit,a.selection,
            str(Path(a.latents)/'config.json'),
            'train/train_step1.py','train/diagnostics.py','model/runtime.py',
            'configs/config.py','utils/hashing.py',a.config,'model/receiver.py','eval/controls.py',
            'model/adapter.py']},
        'freeze': freeze, 'delimiter_init': receiver.delimiter_init_report,
        'resume': None if resume is None else {'checkpoint_sha256': sha256_file(a.resume_epoch),
            'step': start_step, 'completed_epochs': start_epoch, 'optimizer_reset': True,
            'source_checkpoint': str(Path(a.resume_epoch).resolve()),
            'source_training_log_sha256': sha256_file(Path(a.resume_epoch).resolve().parent/'training.jsonl'),
            'inherited_best': history['best'],
            'inherited_best_sha256': sha256_file(history['best_path'])}}
    write_json(str(out/'config.json'), config)
    before = parameter_fingerprint(receiver.base_model)
    write_json(str(out/'base_before.json'), {'sha256': before})
    opt = torch.optim.AdamW(receiver.adapter.parameters(), lr=training['learning_rate'],
        weight_decay=training['weight_decay'], betas=tuple(training['betas']), eps=training['eps'])
    start = time.time()
    def log(rec):
        rec = {'elapsed_s': time.time()-start, **rec}
        with (out/'training.jsonl').open('a') as f:
            f.write(json.dumps(rec)+'\n')
        print(json.dumps(rec), flush=True)
    def save(name, step, val):
        torch.save({'adapter': receiver.adapter.state_dict(), 'step': step, 'selection_nll': val,
            'base': RECEIVER_MODEL, 'revision': RECEIVER_REVISION, 'config': config,
            'optimizer': opt.state_dict(), 'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all()}, out/name)
    if resume is None:
        initial = validate(receiver, selection, out/'selection_0000.jsonl')
        best = {'step': 0, 'nll': initial}
        save('initial.pt', 0, initial); save('best.pt', 0, initial)
        log({'event': 'initial_validation', 'nll': initial})
    else:
        initial = history['initial_nll']
        shutil.copy2(history['initial_path'], out/'initial.pt')
        shutil.copy2(a.resume_epoch, out/f'epoch_{start_epoch:02d}.pt')
        best = dict(history['best'])
        # Copy the selected checkpoint itself, not the current resume weights
        # with a historical score attached to them.
        shutil.copy2(history['best_path'], out/'best.pt')
        log({'event': 'resume', 'step': start_step, 'completed_epochs': start_epoch,
            'nll': resume['selection_nll'], 'best': best, 'optimizer_reset': True})
        del resume
    if probes:
        log(probes.measure(receiver, communication_mix, start_step))
    train_start = time.time(); step = start_step; processed = start_epoch*len(fit); stopped = None
    rng = random.Random(TRAIN_SEED)
    # Replay inexpensive sampling only, preserving the original epoch-2/3 ordering.
    for epoch in range(a.epochs):
        for indices, shift, fractions in epoch_batches(len(fit), group_size, rng, training['latent_fractions']):
            if epoch < start_epoch:
                continue
            chunk = [fit[i] for i in indices]
            donors = chunk[shift:] + chunk[:shift]
            tokens = sum(sum(l != -100 for l in r['rendered'].labels[1:]) for r in chunk)
            weights = schedule(step,total_steps,training)
            for g in opt.param_groups:
                g['lr'] = weights['lr']
            opt.zero_grad(set_to_none=True)
            receiver.adapter.train(); receiver.base_model.train()  # enable checkpointing; Qwen dropout=0
            values = torch.zeros(4); norms = []
            example_weights = []; mean_weights = {'sep': 0., 'align': 0.}
            weighted_loss = 0.
            t0 = time.time()
            for row, donor, fraction in zip(chunk,donors,fractions):
                text = text_message(receiver,row)
                z = adapted(receiver,row)
                donor_raw = align_donor_latent(latent(donor, receiver.adapter.hidden_size), len(text), training['donor_noise_std'])
                zd = receiver.adapter.process_hidden_states(donor_raw)
                mixed = communication_mix(text,z,fraction)
                negative = communication_mix(text,zd,fraction)
                hm,y = supervised_hidden(receiver,row['rendered'],mixed)
                hn,yn = supervised_hidden(receiver,row['rendered'],negative)
                with torch.no_grad():
                    ht,yt = supervised_hidden(receiver,row['rendered'],text)
                assert torch.equal(y,yn) and torch.equal(y,yt)
                loss, parts, coefficients = loss_from_hidden(receiver.base_model.lm_head,hm,hn,ht,y,config=training)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite loss at update {step}, task {row["id"]}')
                weight = len(y)/tokens
                scalars = {k: float(v.detach()) for k, v in coefficients.items()}
                example_weights.append({'id': row['id'], **scalars})
                for key, value in scalars.items():
                    mean_weights[key] += value*weight
                weighted_loss += float(loss.detach())*weight
                if loss.requires_grad:
                    (loss*weight).backward()
                values += parts.detach().cpu()*weight
                norms.append(float(z.detach().norm(dim=-1).mean()))
                del hm,hn,ht,loss,parts,z,zd,mixed,negative,coefficients,donor_raw
            grad = torch.nn.utils.clip_grad_norm_(receiver.adapter.parameters(),training['grad_clip'],error_if_nonfinite=True)
            assert all(p.grad is None for p in receiver.base_model.parameters())
            opt.step(); step += 1; processed += len(chunk)
            log({'event':'update','step':step,'epoch':epoch+1,'examples':processed,'tokens':tokens,
                'ids':[r['id'] for r in chunk], 'donors':[r['id'] for r in donors], 'fractions':fractions,
                'weights':{**weights, **mean_weights}, 'example_weights':example_weights,
                'loss':weighted_loss, 'losses':values.tolist(),'grad_norm_before_clip':float(grad),
                'mean_message_token_norm':sum(norms)/len(norms),'update_s':time.time()-t0,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            if probes:
                log(probes.measure(receiver, communication_mix, step))
            if step % training['checkpoint_interval'] == 0 or probes:
                save(f'update_{step:04d}.pt',step,None)
            stopped = training_stop_reason(step, a.stop_after_updates, time.time()-train_start, a.max_hours)
            if stopped:
                break
        if epoch < start_epoch:
            continue
        val = validate(receiver,selection,out/f'selection_{step:04d}.jsonl')
        save('last.pt',step,val)
        if step == (epoch+1)*math.ceil(len(fit)/group_size):
            save(f'epoch_{epoch+1:02d}.pt',step,val)
        if val < best['nll']:
            best = {'step':step,'nll':val}; save('best.pt',step,val)
        log({'event':'validation','epoch':epoch+1,'step':step,'nll':val,'best':best})
        if stopped:
            break
    after = parameter_fingerprint(receiver.base_model)
    assert before == after, 'frozen Base parameters changed'
    write_json(str(out/'summary.json'), {'updates':step,'examples':processed,'best':best,
        'initial_nll':initial,'last_nll':val,'elapsed_s':time.time()-start,
        'stop_reason':stopped or 'epoch_budget',
        'base_unchanged':before==after,'base_sha256':after,
        'checkpoint_sha256':{p.name:sha256_file(str(p)) for p in out.glob('*.pt')}})


if __name__ == '__main__':
    main()
