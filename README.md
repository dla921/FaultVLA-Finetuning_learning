# FaultVLA-Finetuning_learning

Training paths:
- `server/train.py`: original frozen-feature linear baseline.
- `scripts/train_bridge.py`: GPU finetuning of the actual Pro Policy/Bridge Attention.
- `scripts/check_action_predictions.py`: offline action-error diagnostic for either checkpoint.

See [Policy training instructions](docs/POLICY_TRAINING.md) before switching paths.

Public server package for **SO101_StableGrasp200**: SO-101 arm + AmazingHand,
200 successful simulated cup transfers with thumb side contact.

## Data contract

- Original episode split: train 160 / validation 20 / test 20 (no re-splitting).
- 75,600 front RGB frames, 224x224 at 10 Hz; no wrist camera or depth.
- 378,000 joint observations/commands at 50 Hz; 13 absolute joint targets in radians.
- `rgb_state_index` selects the state before the next five 20 ms commands: target shape `[5,13]`.
- Statistics are recomputed using only the 160 training episodes.
- Full state, cup position and contact diagnostics are not policy inputs.

The four original ZIP files are public Release assets, with sizes and SHA-256 in
`configs/data_release.json`. Large data and model weights are excluded from Git.

## Start on a server

Ubuntu x86_64, Conda, Python 3.10 and a CUDA BF16 GPU are required for VLM features.
Allow at least 20 GB disk and 16 GB RAM (32 GB recommended).

```bash
git clone https://github.com/dla921/FaultVLA-Finetuning_learning.git
cd FaultVLA-Finetuning_learning
conda create -n FaultVLA-Finetuning python=3.10 pip -y
conda activate FaultVLA-Finetuning
bash server/install.sh
python server/restore_data.py
python server/preflight.py --data-only
python server/download_checkpoint.py --model object
python server/preflight.py --model object --inference
python server/train.py --model object --smoke --epochs 1 --output results/separated_smoke
python server/train.py --model object --epochs 20 --output results/separated_full
```

Each output directory must be new. Use tmux for full feature extraction/training.
Training extracts frozen single-camera Object-Pro visual/language features.
**Inference input is front RGB + task text only**: no arm/hand qpos, qvel,
contact forces, simulator object poses or injected-fault IDs are supplied.

The readout has three separate roles:
- Arm: five absolute joint targets at each of five 20 ms steps, clipped to source actuator limits.
- Hand: two bounded synergy commands (`finger_closure`, `thumb_closure`) at each step.
  `HandController` expands them into the eight opposing motor targets. The mapping
  exactly reproduces the original data, including the different thumb target.
  It is sensor-free command generation, not force control or measured fault compensation.
- Visual monitor: RGB-based probabilities for source region, elevated and target region.
  Simulator cup positions provide offline labels only. These are location estimates,
  not grasp/contact verification, success certification or motor-fault diagnoses.

Only the new readout is trained; the VLM remains frozen. This is not full-model/LoRA
finetuning. Train-only normalization and validation selection are retained; test is held out.
The saved `separated_policy.pt` is loaded by
`faultvla.stablegrasp_policy.SeparatedPolicy(checkpoint, key, trained)`:

```python
result = policy.predict(front_rgb_uint8_224x224, task_text)
# result['arm_targets_rad']: [5,5] absolute radians, 50 Hz
# result['hand_commands']: [5,2] synergy commands in [0,1]
# result['hand_targets_rad']: [5,8], from the separate hand controller
# result['visual_probabilities']: camera-derived location estimates
```

For a single-frame inference check after training:

```bash
python server/predict.py --trained results/separated_full/separated_policy.pt \
  --image data/SO101_StableGrasp200/episodes/episode_000/rgb/000000.png \
  --task "Pick up the cup and place it on the other table." \
  --output results/separated_prediction.json
```

This writes commands to JSON; it does not move a robot.

The arm and hand chunks must execute synchronously at 50 Hz; acquire a fresh RGB
frame and replan every 100 ms. This is the dataset timing requirement, not a measured
real-time guarantee. A real controller/simulator must implement and benchmark this loop.
`scripts/evaluate_separated.py` provides a MuJoCo closed-loop runner. Physics replay
and a short smoke-policy connection were tested locally; a fully trained policy
has not yet been evaluated. See docs/MUJOCO_EVALUATION.md.
There is no automatic recovery supervisor or reliable fault detector trained from
these normal-only demonstrations. Add failure/recovery data and temporal visual
monitoring before claiming fault-tolerant operation.

**Retrain after this change.** Earlier `joint_readout.pt` checkpoints are incompatible
and are rejected; use a fresh output directory (for example `results/separated_smoke`
and `results/separated_full`). Existing data and checkpoints are preserved.

## Validation and source assets

```bash
python -m unittest discover -s tests -v
python data/SO101_StableGrasp200/verify_final_dataset.py
```

The supplied verifier checks original model/controller hashes, all PNGs, joint
limits, temporal alignment and contact-quality records. The original simulation
was collected with MuJoCo 3.3.7; the existing VLA environment uses 3.17.0. Dataset
verification does not establish replay equivalence across these versions.
For exact source simulation reproduction, use a separate environment with
`data/SO101_StableGrasp200/requirements.txt`.

- `reports/stablegrasp200_validation.json`: locally rerun full dataset checks.
- `reports/separated_smoke_training.json`: current camera-only separated GPU smoke result.
- `reports/stablegrasp_smoke_training.json`: historical 13-axis joint-readout result.
- `data/SO101_StableGrasp200/README.md`: supplied dataset description and limits.
- Dataset `provenance/`: original SO-101 and AmazingHand sources and licenses.
- `VLA-Adapter/`: upstream source snapshot, MIT LICENSE retained.

The previous wide200 dataset, four local tar.gz archives and old dataset GitHub
Release have been deleted. Only StableGrasp200 and its four ZIP archives remain
in the project data/archive directories. Historical code, reports and training
results are preserved; they are not used by the current training path. Older Cartesian adapter/evaluation code remains as legacy
reference and is **not compatible** with this new joint readout. Use `server/train.py`
for the current dataset. Full 200-episode training, closed-loop simulated task
success, fault recovery and real-robot performance have not been measured.

Data and models are simulated experiment assets supplied by the user. The collection
covers a narrow initial-position range, not broad real-world generalization.

## Policy finetuning and baseline diagnostics

The original server/train.py path is a frozen-feature linear baseline. For offline
action prediction errors and GPU training of the actual 24-block Pro Policy/Bridge
Attention, follow [docs/POLICY_TRAINING.md](docs/POLICY_TRAINING.md). The new
scripts/train_bridge.py path keeps the VLM frozen, does not use LoRA or joint
sensor inputs, and preserves separate hand control. Existing baseline checkpoints
remain supported. These are adapted experiments, not paper-exact reproduction.

## Optional visual hand judge

[docs/VISUAL_HAND_JUDGE.md](docs/VISUAL_HAND_JUDGE.md) explains a separately
trained camera-based close/release decision supplement. It is opt-in via
--hand-judge, uses no simulator ground-truth inputs, and does not modify arm
weights. It is experimental and does not guarantee grasp readiness or task success.

## Causal command-history Policy

[docs/COORDINATED_POLICY.md](docs/COORDINATED_POLICY.md) documents a separate
GPU Policy experiment conditioned on previously issued commands and command-derived
hand stage. It preserves existing checkpoints and uses no measured joint/object
state. New training entrypoint: scripts/train_coordinated.py.
