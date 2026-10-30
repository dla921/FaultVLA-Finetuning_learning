# Camera-only hand close/release decision (experimental)

This is an optional supplement to the saved arm policy. Arm weights and arm
commands are unchanged. No retraining of the existing arm checkpoint is required.
A separate classifier learns approach / ready_to_close / carry / release from
frozen VLM image+instruction features. No qpos, cup position, contacts, injected
fault IDs or elapsed task time enters the classifier or hand supervisor.

## Train a separate judge

```bash
python scripts/train_hand_judge.py --model object --epochs 20 --stride 5 \
 --output results/hand_judge_01
```

Use tmux for feature extraction. Default: every fifth RGB (2Hz observations), all
160 training / 20 validation episodes, test untouched. Training phases are offline
annotation labels only. Approach includes the alignment movement. Only
02_preclose_settle and 03_close label ready_to_close; lift/check/transfer/place
label carry; release/retreat/final label release. A settle interval inherits the
preceding source phase label. Source phases are proxies, not measured grasp
readiness or force. Normal-only labels do not validate fault detection/recovery.

Features are extracted on GPU with the frozen source Object-Pro VLM. A small MLP
is trained on GPU with AdamW lr1e-4, weight decay .01, batch64, train-frequency
class weights, train-only feature mean/std, gradient clip1, and validation loss
selection. hand_judge.pt is separate from bridge_policy.pt. metrics.json records
history and a 4x4 best-checkpoint confusion matrix (rows=true, columns=predicted;
class order is recorded). features.pt keeps local extracted features/labels and
identities; no raw simulator telemetry is used for inference.

For a wiring check first, add --smoke --epochs 2 into a NEW output directory. It
uses one frame per stage from one train and one validation episode; smoke accuracy
is not evidence of useful readiness detection.

## Compare on the same development episode

```bash
PYOPENGL_PLATFORM=egl python scripts/evaluate_separated.py --mode policy \
 --trained results/bridge_normal_01/bridge_policy.pt \
 --hand-judge results/hand_judge_01/hand_judge.pt \
 --judge-threshold .8 --judge-confirmations 3 --judge-hand-speed 1 \
 --episode episode_081 --video \
 --output results/bridge_normal_01/mujoco_hand_judge_01
```

No global arm/hand rate flags here, to match the previous unfiltered Bridge run.
Until takeover, the baseline's approach hand-aperture commands are retained. This
is a close/release handoff supplement, not a fully independent approach controller.
If ready_to_close probability is >=.8 for three consecutive 10Hz camera frames,
the supervisor sends the dataset's closed targets [.5,-.5] for three finger pairs
and [-.3,.3] for the thumb. The command ramps at 1rad/s at 50Hz from the previously
issued command, never from measured joint position. Then it keeps the close
command. The label holding means commanded hold, not confirmed grasp.

After at least five holding camera updates, three confident release frames trigger
opening to [-.8,.8] for all pairs, also ramped. There is no elapsed-task schedule
or forced close timeout. A single/uncertain frame does not trigger close. The
confidence threshold is a starting heuristic, not a calibrated physical guarantee.
A misclassified pose can close prematurely; an unaligned pose may never trigger.
This is a simulation experiment, not validated real-hardware control.

summary.json.visual_hand_supervisor records events and final state. Each camera
update also logs visual_hand_decision probabilities/state plus the VLA hand
proposal in trace.jsonl. Geometry/contact diagnostics remain evaluation-only and
are never passed to the hand judge. Keep camera feature extraction identical at
train/inference; reuse the policy's current-frame features with no second VLM pass.

Interpretation:
- No close_command_triggered: inspect readiness probabilities and arm alignment;
  do not lower the threshold just to force closure.
- Close triggered but no lift: inspect contact/geometry and synchronization; the
  judge does not repair the arm path or stop its motion during closure.
- A successful single development episode is not a success rate. Evaluate reserved
  conditions before fault-tolerance claims. Keep identical hand-judge settings in
  algorithm-on/off fault-tolerance comparisons.

Without --hand-judge the original evaluation path is unchanged. Existing weights
and prior runs are preserved. No LoRA, recovery supervisor or force feedback was added.
