# FaultVLA-Finetuning_learning

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
python server/train.py --model object --smoke --epochs 1 --output results/stablegrasp_smoke
python server/train.py --model object --epochs 20 --output results/stablegrasp_full
```

Each output directory must be new. Use tmux for full feature extraction/training.
Training extracts frozen single-camera Object-Pro visual/language features,
concatenates normalized 13-joint position, and learns a new linear `[5,13]` readout.
The original 7D action decoder and 8D proprioception projector are not trained or
used as robot commands. This is a frozen-feature experiment, not full VLA/LoRA finetuning.
Only train/validation are used; the test split is held out.
The saved `joint_readout.pt` contains weights and train-only normalization.
`faultvla.stablegrasp_policy.JointPolicy` loads it with the same base checkpoint.
Its predictions are absolute radians, not deltas; real hardware requires a separate
controller with joint limits, timing and safety checks. No hardware evaluation was performed.

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
- `reports/stablegrasp_smoke_training.json`: small real-GPU training result.
- `data/SO101_StableGrasp200/README.md`: supplied dataset description and limits.
- Dataset `provenance/`: original SO-101 and AmazingHand sources and licenses.
- `VLA-Adapter/`: upstream source snapshot, MIT LICENSE retained.

The earlier wide200 dataset, release archives and historical reports are preserved
pending explicit cleanup approval. They are not used by the current training path. Older Cartesian adapter/evaluation code remains as legacy
reference and is **not compatible** with this new joint readout. Use `server/train.py`
for the current dataset. Full 200-episode training, closed-loop simulated task
success, fault recovery and real-robot performance have not been measured.

Data and models are simulated experiment assets supplied by the user. The collection
covers a narrow initial-position range, not broad real-world generalization.
