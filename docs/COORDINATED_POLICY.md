# Command-history-conditioned Policy experiment

This is a separate model, not a fix guaranteed to solve grasp/transport. Original
linear and Bridge policies, arm weights and judge checkpoints are preserved.

## Inputs and causal boundary

The new policy receives RGB/task plus 12 values: previous issued arm targets (5),
previous issued hand synergies (2), a command-derived four-state one-hot and a
history-valid bit. Previous commands are normalized using train-only action stats.
The first prediction has no history (zero numeric fields, approach state, valid=0).
States are approach_command / closing_command / closed_command / opening_command.
Closed means targets were sent (main >=.43rad and thumb near -.3rad), not that
fingers reached them, contact occurred or a cup was held. The state machine uses
only issued targets and their changes; it never consumes source phase, measured
qpos/qvel, simulator cup pose/contact, fault IDs, task clock or success labels.

The context MLP produces a conditioning token for all 24 Pro Bridge blocks. VLM
weights stay frozen; the active Policy and new context projector train on GPU.
No LoRA, image-history encoder, force controller, fixed lift schedule or automatic
arm trajectory is introduced. A separate visual hand judge remains optional.
Its internal predicted phase is not passed as an oracle: the shared context stage
is reconstructed from actual outgoing hand commands, including judge overrides.

Training computes context from actions strictly before the current frame's index,
including intervening 50Hz commands even when training RGB is subsampled/shuffled.
Runtime updates context with each post-limiter applied target, not raw proposed
commands. The same builder/state rules are used. Training uses expert command
history (teacher forcing); runtime uses its own outputs, so distribution shift
and correction-data needs remain. A command-state token is not grasp confirmation.

## Warm-start from your trained Bridge policy

```bash
python scripts/train_coordinated.py --model object \
 --init-policy results/bridge_normal_01/bridge_policy.pt \
 --smoke --epochs 1 --grad-accumulation 2 \
 --output results/context_smoke_01
```

This loads existing Bridge network/output/visual weights and adds a new randomly
initialized context projection. Source normalization must match. Omitting
--init-policy starts from the source Object-Pro policy with a reset output layer.
No original checkpoint is overwritten. Smoke uses three spread frames per split
from one train/validation episode and verifies context/query/gate/output updates;
it is not a performance estimate.

After diagnostic smoke, a starting larger run (tmux recommended):

```bash
python -u scripts/train_coordinated.py --model object \
 --init-policy results/bridge_normal_01/bridge_policy.pt \
 --epochs 10 --stride 5 --learning-rate 1e-4 --grad-accumulation 16 \
 --output results/context_normal_01 > context_normal_01.log 2>&1
```

Loss, scheduler and split rules match train_bridge.py. The new checkpoint is
context_policy.pt; lowest validation loss selects it. Metrics explicitly check
context projector weight updates. Test data never participates in training.
Default stride5 means 2Hz training RGB with original five-command 50Hz targets.
Settings are starting experiment values, not proven optimal or paper-exact.

## Evaluation and diagnostic starts

```bash
PYOPENGL_PLATFORM=egl python scripts/evaluate_separated.py --mode policy \
 --trained results/context_normal_01/context_policy.pt \
 --hand-judge results/hand_judge_01/hand_judge.pt --episode episode_081 --video \
 --output results/context_normal_01/mujoco_full_01
```

Use the same judge/filter settings as baseline for comparisons. The new checkpoint
contract auto-selects the context-aware loader. Every 20ms outgoing command is fed
back; policy_command_context is logged at each camera update. Full task_success
still requires a real learned-policy rollout. Do not infer success from loss.
For --start-phase 02_preclose_settle, only already executed expert-prefix commands
seed context, then actual policy/judge/filter commands take over. Mid-start
success remains a diagnostic and excluded from full-task success.

The offline action diagnostic auto-uses past expert commands for context policies;
its note explicitly says teacher-forced history. Single-frame server/predict.py
starts with empty history unless --issued-history supplies a JSON list of past
13D commands. Programmatic users must call observe_issued_command only after
sending each actual command and reset_command_context for every fresh episode.

If the model still stalls, investigate distribution shift/corrective demonstrations
and temporal perception; adding internal command history alone is not evidence
of closed-loop robustness, grasp stability or fault tolerance.
