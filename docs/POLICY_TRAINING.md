# Diagnose first, then train the Pro Policy

The original linear baseline is retained. Do not overwrite its results.

## Offline action prediction errors

```bash
python scripts/check_action_predictions.py \
 --trained results/normal_baseline_01/separated_policy.pt \
 --split validation --samples-per-episode 20 \
 --output results/normal_baseline_01/offline_validation_01
```

Default: 20 evenly spaced RGB frames from each of 20 validation episodes = 400
frames. Each target has five 50Hz commands. summary.json reports per-joint arm MAE
in radians/degrees, raw errors, applied joint-clipped errors, hand-command errors,
hand motor errors, horizon errors, clipping counts and visual-location accuracy.
predictions.jsonl contains every prediction and target. This uses recorded images,
not generated rollouts, and does not update weights. There is no command-rate filter
in this offline diagnostic. Compare with closed-loop outcomes; no single MAE cutoff
proves a cause or guarantees successful grasp. Use validation for development;
reserve test cases not already used for tuning for final reporting.

## What the new learner actually trains

scripts/train_bridge.py uses the upstream 24-block Pro MLPResNet and Bridge
Attention, initialized from the Object/Spatial-Pro policy. It retains the full
25-layer visual/action tokens rather than mean-pooling them to 896 dimensions.
All active Policy parameters, the output projection, visual classifier and a
learned constant conditioning token are trainable on GPU. Unused upstream FiLM
parameters are frozen. No qpos, qvel, contact forces or fault IDs enter the model.
The source proprio projector is replaced by a learned constant token, not invented
sensor measurements. The source 8-slot decoder is retained internally and only
its first five outputs are supervised/executed as the five 20ms commands.
The output projection is reset because SO101 arm targets and hand synergies have
different semantics from the source Cartesian action labels.

The VLM stays frozen and LoRA is not implemented/enabled by this trainer. This is
an adapted camera-only Policy finetuning experiment, not paper-exact reproduction.
A small vendor patch lets VLA_ROBOT_PLATFORM explicitly pin the pretrained LIBERO
dimensions; otherwise upstream guesses from filenames containing bridge/aloha.

## Smoke test before full training

```bash
python scripts/train_bridge.py --model object --smoke --epochs 1 \
 --grad-accumulation 2 --output results/bridge_smoke_01
```

Uses three evenly spaced frames from one train and one validation episode.
It checks actual query, bridge gate and output weight changes and absence of VLM
gradients. Smoke loss/accuracy is not a task-performance estimate.

## Larger run (use tmux)

```bash
python -u scripts/train_bridge.py --model object --epochs 10 --stride 5 \
 --learning-rate 1e-4 --grad-accumulation 16 \
 --output results/bridge_normal_01 > bridge_normal_01.log 2>&1
```

Default stride 5 subsamples every fifth RGB frame (2Hz training observations) but
keeps every target's original five consecutive 50Hz commands. Use --stride 1 for
all RGB frames. Training order is shuffled reproducibly by epoch; validation is
fixed. Statistics always use all 160 training episodes. No test images are used.
Microbatch=1 on GPU; accumulation=16 gives up to 16 samples per optimizer update.
AdamW, weight decay .01, 10% warmup + cosine decay, gradient clipping 1, and L1
normalized action loss + .1 visual cross entropy + .1 supervised within-chunk
command-difference loss. The last loss matches expert differences rather than
forcing every motion to zero. Values are starting settings, not validated optimal
hyperparameters. They differ from the original linear learner and the paper.

Full multilayer features are extracted online every epoch; no huge token cache
is saved. Expect substantially more GPU compute/VRAM than the linear probe.
A small local GPU smoke run does not establish full A6000 throughput. Check logs
and nvidia-smi before committing to a long run. No mid-epoch resume is supported;
settings.json and metrics.json record settings/update counts; the lowest validation
loss checkpoint is atomically saved as bridge_policy.pt after each epoch.

## Evaluate saved Policy weights

```bash
PYOPENGL_PLATFORM=egl python scripts/evaluate_separated.py --mode policy \
 --trained results/bridge_normal_01/bridge_policy.pt --episode episode_081 \
 --arm-max-speed 1 --hand-max-speed 2 --video \
 --output results/bridge_normal_01/mujoco_dev_01
```

The loader dispatches by checkpoint contract. Old linear separated_policy.pt files
still work; new Bridge files use the trained attention Policy. Keep the same command
filter for baseline comparisons. Source replay is not learned policy evaluation.
No full-trained Bridge checkpoint or task success has been validated yet. Inspect
summary.json and rollout.gif, then repeat on reserved evaluation conditions.
