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


VISUAL_STATES = ('source_region','elevated','target_region')


def visual_labels(cup_xyz):
    """Offline simulation positions supply labels only, never inference inputs.
    These are location states, not contact, success or motor-fault diagnoses.
    """
    xyz=np.asarray(cup_xyz,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite cup trajectory')
    source=xyz[0];target=xyz[-1]
    if np.linalg.norm(target[:2]-source[:2])<.02:raise ValueError('No distinct destination')
    closer_target=np.linalg.norm(xyz[:,:2]-target[:2],axis=1)<np.linalg.norm(xyz[:,:2]-source[:2],axis=1)
    labels=np.where(closer_target,2,0)
    labels[xyz[:,2]>max(source[2],target[2])+.015]=1
    return labels


def separated_samples(root,split,max_episodes=None,max_samples=None):
    """Camera-only inputs; arm labels and hand synergies remain synchronized."""
    from faultvla.hand_controller import HandController
    root=Path(root)
    for episode in split_ids(root,split)[:max_episodes]:
        ep=root/'episodes'/episode
        meta=json.loads((ep/'metadata.json').read_text())
        if not (meta['accepted'] and meta['images_saved'] and meta['task_success']):raise ValueError(episode)
        with np.load(ep/'data.npz') as z:
            labels=visual_labels(z['cup_xyz'])
            for index in z['rgb_state_index'][:max_samples]:
                index=int(index);action=z['action'][index:index+5].copy()
                if action.shape!=(5,13) or not np.isfinite(action).all():raise ValueError('Action chunk')
                hand=HandController.encode(action[:,5:])
                with Image.open(ep/'rgb'/f'{index:06d}.png') as im:rgb=np.array(im.convert('RGB'))
                if rgb.shape!=(224,224,3):raise ValueError('RGB shape')
                yield dict(episode=episode,index=index,rgb=rgb,task=meta['instruction'],
                           action=np.concatenate((action[:,:5],hand),axis=1),visual_label=int(labels[index]))


def separated_statistics(root):
    from faultvla.hand_controller import HandController
    arrays=[]
    for episode in split_ids(root,'train'):
        with np.load(Path(root)/'episodes'/episode/'data.npz') as z:
            action=z['action']
            if action.ndim!=2 or action.shape[1]!=13 or not np.isfinite(action).all():raise ValueError('Action shape')
            arrays.append(np.concatenate((action[:,:5],HandController.encode(action[:,5:])),axis=1))
    x=np.concatenate(arrays)
    return dict(mean=x.mean(0).tolist(),std=np.maximum(x.std(0),1e-6).tolist())
