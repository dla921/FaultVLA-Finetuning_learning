"""Native single-camera, 13-joint, five-command data contract."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def split_ids(root, split):
    manifest = json.loads((Path(root)/'manifest_200.json').read_text())['splits']
    ids = sum(manifest.values(), [])
    if len(ids) != len(set(ids)) or {k:len(v) for k,v in manifest.items()} != dict(train=160, validation=20, test=20):
        raise ValueError('Expected disjoint 160/20/20 episode split')
    return manifest[split]


def statistics(root):
    arrays = {key:[] for key in ('qpos','action')}
    for episode in split_ids(root, 'train'):
        with np.load(Path(root)/'episodes'/episode/'data.npz') as z:
            for key in arrays: arrays[key].append(z[key])
    result = {}
    for key, values in arrays.items():
        x = np.concatenate(values)
        if x.shape[1] != 13 or not np.isfinite(x).all(): raise ValueError(key)
        result[key] = dict(mean=x.mean(0).tolist(), std=np.maximum(x.std(0),1e-6).tolist())
    return result


def samples(root, split, max_episodes=None, max_samples=None):
    root = Path(root)
    for episode in split_ids(root,split)[:max_episodes]:
        ep = root/'episodes'/episode
        meta = json.loads((ep/'metadata.json').read_text())
        if not (meta['accepted'] and meta['images_saved'] and meta['task_success']): raise ValueError(episode)
        with np.load(ep/'data.npz') as z:
            for index in z['rgb_state_index'][:max_samples]:
                index = int(index); action=z['action'][index:index+5].copy()
                state=z['qpos'][index].copy()
                if action.shape != (5,13) or state.shape != (13,) or not np.isfinite(action).all() or not np.isfinite(state).all():
                    raise ValueError('Invalid native action/state')
                with Image.open(ep/'rgb'/f'{index:06d}.png') as im: rgb=np.array(im.convert('RGB'))
                if rgb.shape != (224,224,3): raise ValueError('RGB shape')
                yield dict(episode=episode,index=index,rgb=rgb,state=state,action=action,task=meta['instruction'])
