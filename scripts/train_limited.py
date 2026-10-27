"""Train only the final action projection on real, frozen VLA features.

This is a deliberately restricted linear-probe experiment, not full-head/LoRA training.
No expert trajectory is used at inference. No backbone or base checkpoint is modified.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'VLA-Adapter')]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
from PIL import Image
import torch
from faultvla.limited_training import (STATS_KEY, training_statistics, examples,
                                      normalize_actions, denormalize_actions)
from faultvla.policy import Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/LIBERO-Object-Pro')
    parser.add_argument('--data', type=Path, default=ROOT/'data/cartesian_demos_v2_wide200')
    parser.add_argument('--base-unnorm-key', default='libero_object_no_noops')
    parser.add_argument('--all-episodes', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-episodes', nargs='+', default=['train_0000'])
    parser.add_argument('--validation-episodes', nargs='+', default=['validation_0000'])
    parser.add_argument('--max-samples-per-episode', type=int, default=16)
    parser.add_argument('--all-samples', action='store_true')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.all_episodes:
        args.train_episodes = [e['id'] for e in json.loads((args.data/'train.json').read_text())]
        args.validation_episodes = [e['id'] for e in json.loads((args.data/'validation.json').read_text())]
    if args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError('Positive epochs and learning rate required')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment needs the existing CUDA inference environment')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    (args.output/'settings.json').write_text(json.dumps(settings, indent=2))
    stats, stats_episodes = training_statistics(args.data)
    (args.output/'dataset_statistics.json').write_text(json.dumps(stats, indent=2))
    maximum = None if args.all_samples else args.max_samples_per_episode
    records = {split: list(examples(args.data, split, selected, args.stride, maximum))
               for split,selected in [('train',args.train_episodes), ('validation',args.validation_episodes)]}
    if not all(records.values()):
        raise ValueError('Both train and validation must contain complete action chunks')
    torch.cuda.reset_peak_memory_stats()
    policy = Policy(args.checkpoint, args.base_unnorm_key, replan_actions=8)
    policy.stats = stats; policy.model.norm_stats = stats; policy.cfg.unnorm_key = STATS_KEY
    for module in [policy.model, policy.head, policy.proprio]:
        module.eval().requires_grad_(False)
    capture = {}
    def feature_hook(module, inputs):
        capture['features'] = inputs[0].detach().cpu()
    hook = policy.head.model.fc2.register_forward_pre_hook(feature_hook)
    cache = {}
    try:
        for split, samples in records.items():
            features, targets, identities = [], [], []
            for k,sample in enumerate(samples):
                row = sample['row']; ep = sample['directory']
                obs = dict(full_image=np.array(Image.open(ep/row['images']['front']).convert('RGB')),
                    wrist_image=np.array(Image.open(ep/row['images']['wrist']).convert('RGB')),
                    state=np.asarray(row['state'], dtype=np.float32))
                capture.clear()
                policy.predict(obs, row['task'])
                # Clone OUTSIDE inference_mode: normal tensors are needed for autograd.
                features.append(capture['features'].clone().float().squeeze(0))
                targets.append(torch.from_numpy(normalize_actions(sample['actions'], stats[STATS_KEY]['action'])))
                identities.append(dict(episode=sample['episode'], index=sample['index']))
                if k == 0 or (k+1) % 10 == 0 or k+1 == len(samples):
                    print(f'Features {split}: {k+1}/{len(samples)}', flush=True)
            cache[split] = dict(features=torch.stack(features), targets=torch.stack(targets), identities=identities)
    finally:
        hook.remove()
    torch.save(cache, args.output/'features.pt')
    feature_peak = dict(allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    layer = copy.deepcopy(policy.head.model.fc2).float().requires_grad_(True).train()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=args.learning_rate, weight_decay=0)
    initial_weight = layer.weight.detach().clone()
    def evaluate(split):
        data = cache[split]
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(data['features']), 256):
                prediction = layer(data['features'][start:start+256].cuda()).cpu()
                target = data['targets'][start:start+256]
                total += float(torch.nn.functional.l1_loss(prediction, target, reduction='sum'))
                count += target.numel()
        return total / count
    initial = {split:evaluate(split) for split in cache}
    best_loss = initial['validation']; best = copy.deepcopy(layer.state_dict()); best_epoch = 0
    history = [dict(epoch=0, **initial)]
    for epoch in range(1,args.epochs+1):
        for idx in torch.randperm(len(cache['train']['features'])).tolist():
            optimizer.zero_grad(set_to_none=True)
            # Actual micro-batch size = 1; loss uses eight consecutive actions.
            output = layer(cache['train']['features'][idx:idx+1].cuda())
            loss = torch.nn.functional.l1_loss(output, cache['train']['targets'][idx:idx+1].cuda())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in layer.parameters()):
                raise RuntimeError('Missing/nonfinite output-layer gradients')
            torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
            optimizer.step()
        metrics = dict(epoch=epoch, **{split:evaluate(split) for split in cache})
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics['validation'] < best_loss:
            best_loss = metrics['validation']; best = copy.deepcopy(layer.state_dict()); best_epoch = epoch
    changed = float((layer.weight-initial_weight).abs().max().item())
    assert changed > 0, 'Optimizer did not update parameters'
    assert all(p.grad is None for m in [policy.model,policy.head,policy.proprio] for p in m.parameters())
    torch.save(dict(last=layer.state_dict(),best=best,optimizer=optimizer.state_dict(),best_epoch=best_epoch),
               args.output/'output_layer_training.pt')
    # Export an inference-compatible checkpoint; large frozen components are read-only symlinks.
    exported = args.output/'checkpoint'; exported.mkdir()
    for file in args.checkpoint.resolve().iterdir():
        if file.is_file() and file.name not in ['action_head--checkpoint.pt','dataset_statistics.json']:
            (exported/file.name).symlink_to(file)
    (exported/'dataset_statistics.json').write_text(json.dumps(stats, indent=2))
    policy.head.model.fc2.load_state_dict(best)
    torch.save({k:v.detach().cpu() for k,v in policy.head.state_dict().items()}, exported/'action_head--checkpoint.pt')
    # Verify saved BF16 projection, and full online inference on one real observation.
    state = torch.load(exported/'action_head--checkpoint.pt', map_location='cpu', weights_only=True)
    policy.head.load_state_dict(state)
    sample = records['validation'][0]; row=sample['row']; ep=sample['directory']
    obs=dict(full_image=np.array(Image.open(ep/row['images']['front']).convert('RGB')),
             wrist_image=np.array(Image.open(ep/row['images']['wrist']).convert('RGB')),
             state=np.array(row['state'],dtype=np.float32))
    online = policy.predict(obs,row['task'])
    with torch.no_grad():
        expected_norm = policy.head.model.fc2(cache['validation']['features'][0:1].cuda().to(torch.bfloat16)).float().cpu().numpy()[0]
    expected = denormalize_actions(expected_norm, stats[STATS_KEY]['action'])
    np.testing.assert_allclose(online, expected, rtol=1e-3, atol=1e-4)
    cfg = json.loads((ROOT/'configs/so101_vla.json').read_text(encoding='utf-8-sig'))
    cfg.update(checkpoint=str(exported.resolve()),unnorm_key=STATS_KEY,
               checkpoint_repo='local/output-layer-only',checkpoint_revision=None)
    (args.output/'inference_config.json').write_text(json.dumps(cfg,indent=2))
    (args.output/'history.json').write_text(json.dumps(history,indent=2))
    report = dict(mode='frozen_model_and_head_except_final_fc2', trainable_parameters=sum(p.numel() for p in layer.parameters()),
        batch_size=1,chunk_length=8,normalization_source='train split only',statistics_episodes=stats_episodes,
        examples={s:len(v['features']) for s,v in cache.items()},initial_losses=initial,final_losses=history[-1],
        best_validation_epoch=best_epoch, best_validation_loss=best_loss,
        max_weight_change=changed,frozen_parameters_received_gradients=False,
        feature_peak=feature_peak,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, gpu=torch.cuda.get_device_name(),
        full_inference_export_check=True,rollout_success=None,
        elapsed_seconds=time.monotonic()-started,checkpoint=str(exported.resolve()),
        note='Smoke/linear-probe result, not full-head fine-tuning or closed-loop task success.')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
