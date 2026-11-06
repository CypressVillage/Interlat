"""Fixed-fit-task probes that preserve training modes and random-number state."""
import torch
import torch.nn.functional as F

from model.runtime import adapted, text_message, nll


class TrainingProbes:
    def __init__(self, receiver, fit, specification):
        by_id = {row['id']: row for row in fit}
        if not specification or len({item['id'] for item in specification}) != len(specification):
            raise ValueError('diagnostic probes require nonempty, unique fit IDs')
        self.tasks = []
        for item in specification:
            fraction = item['latent_fraction']
            if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
                raise ValueError('probe latent_fraction must be in [0,1]')
            self.tasks.append((by_id[item['id']], fraction))
        self.projection = receiver.adapter.adaptive_proj.proj[-1]
        self.initial_projection = {
            name: value.detach().clone() for name, value in self.projection.named_parameters()
        }
        self.initial_latents = {}

    @torch.no_grad()
    def measure(self, receiver, mix, step):
        modes = {module: module.training for module in receiver.modules()}
        device = next(receiver.adapter.parameters()).device
        devices = [device.index] if device.type == 'cuda' else []
        try:
            # Diagnostic forwards must not change later dropout/noise draws.
            with torch.random.fork_rng(devices=devices):
                receiver.eval()
                records = []
                for row, fraction in self.tasks:
                    z = adapted(receiver, row)
                    if row['id'] not in self.initial_latents:
                        self.initial_latents[row['id']] = z.detach().clone()
                    original = self.initial_latents[row['id']]
                    text = text_message(receiver, row)
                    pure = nll(receiver, row, z)
                    mixed = nll(receiver, row, mix(text, z, fraction))
                    records.append({
                        'id': row['id'], 'latent_fraction': fraction, 'tokens': pure['tokens'],
                        'pure_latent_nll': pure['nll'], 'mixed_nll': mixed['nll'],
                        'mean_message_token_norm': float(z.norm(dim=-1).mean()),
                        'relative_message_change': float((z-original).norm()/original.norm().clamp_min(1e-12)),
                        'message_cosine_to_initial': float(F.cosine_similarity(z, original, dim=-1).mean()),
                    })
                initial_sq, delta_sq, max_delta = 0., 0., 0.
                for name, value in self.projection.named_parameters():
                    original = self.initial_projection[name]
                    delta = value-original
                    initial_sq += float(original.square().sum())
                    delta_sq += float(delta.square().sum())
                    max_delta = max(max_delta, float(delta.abs().max()))
                tokens = sum(r['tokens'] for r in records)
                return {
                    'event': 'diagnostic_probe', 'step': step,
                    'scope': 'Fixed fit tasks, teacher-forced, eval mode; not checkpoint selection',
                    'pure_latent_nll': sum(r['pure_latent_nll']*r['tokens'] for r in records)/tokens,
                    'mixed_nll': sum(r['mixed_nll']*r['tokens'] for r in records)/tokens,
                    'projection_delta_norm': delta_sq**.5,
                    'projection_relative_change': (delta_sq/max(initial_sq, 1e-24))**.5,
                    'projection_max_abs_change': max_delta, 'records': records,
                }
        finally:
            for module, training in modes.items():
                module.training = training
