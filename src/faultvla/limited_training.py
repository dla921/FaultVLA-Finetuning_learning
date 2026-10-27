"""Episode-safe data preparation for frozen-feature/output-layer experiments."""
import json
from pathlib import Path
import numpy as np

STATS_KEY = 'faultvla_normal_v1'
CHUNK = 8


def load_split(base, split):
    base = Path(base)
    entries = json.loads((base / (split + '.json')).read_text())
    for entry in entries:
        if entry['split'] != split:
            raise ValueError('Split mismatch')
        ep = base / entry['path']
        meta = json.loads((ep / 'metadata.json').read_text())
        if not meta['success'] or meta['error'] or meta['rejected'] or meta['clipped']:
            raise ValueError('Only validated successful demonstrations are supported')
    return entries


def bounds(values, mask):
    low, high = np.quantile(values, [.01, .99], axis=0)
    # Keep stationary dimensions centered and avoid a singular normalization.
    constant = high - low < 1e-6
    midpoint = (high + low) / 2
    low = np.where(constant, midpoint - 1e-6, low)
    high = np.where(constant, midpoint + 1e-6, high)
    return dict(q01=low.tolist(), q99=high.tolist(), min=values.min(0).tolist(),
                max=values.max(0).tolist(), mean=values.mean(0).tolist(),
                std=values.std(0).tolist(), mask=list(mask))


def training_statistics(base):
    base = Path(base)
    entries = load_split(base, 'train')
    actions, states = [], []
    for entry in entries:
        with np.load(base / entry['path'] / 'trajectory.npz') as data:
            actions.append(data['actions_7d']); states.append(data['state_8d'])
    return {STATS_KEY: dict(action=bounds(np.concatenate(actions), [True]*6 + [False]),
        proprio=bounds(np.concatenate(states), [True]*8),
        num_trajectories=len(entries), num_transitions=sum(map(len, actions)))}, [e['id'] for e in entries]


def normalize_actions(actions, stats, clip=True):
    low, high = np.array(stats['q01']), np.array(stats['q99'])
    normalized = 2 * (actions - low) / (high - low + 1e-8) - 1
    if clip:
        normalized = np.clip(normalized, -1, 1)
    return np.where(stats['mask'], normalized, actions).astype(np.float32)


def denormalize_actions(actions, stats):
    low, high = np.array(stats['q01']), np.array(stats['q99'])
    return np.where(stats['mask'], .5*(actions+1)*(high-low+1e-8)+low, actions)


def chunk_indices(length, stride=1, maximum=None):
    if stride < 1 or (maximum is not None and maximum < 1):
        raise ValueError('Stride and optional sample limit must be positive')
    indices = list(range(0, length - CHUNK + 1, stride))
    if maximum is not None and len(indices) > maximum:
        # Cover the whole task instead of keeping only its initial approach.
        indices = [indices[i] for i in np.linspace(0, len(indices)-1, maximum, dtype=int)]
    return indices


def examples(base, split, selected=None, stride=1, maximum=None):
    base = Path(base)
    entries = load_split(base, split)
    if selected:
        missing = set(selected) - {e['id'] for e in entries}
        if missing:
            raise ValueError(f'Episodes not in {split}: {sorted(missing)}')
        entries = [e for e in entries if e['id'] in selected]
    for entry in entries:
        ep = base / entry['path']
        rows = [json.loads(line) for line in (ep / 'observations.jsonl').read_text().splitlines()]
        with np.load(ep / 'trajectory.npz') as data:
            actions = data['actions_7d'].copy()
        for i in chunk_indices(len(rows), stride, maximum):
            row = rows[i]
            yield dict(episode=entry['id'], index=i, row=row, directory=ep,
                       actions=actions[i:i+CHUNK].copy())
