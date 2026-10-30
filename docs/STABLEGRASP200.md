# StableGrasp200 migration

Current workflow: README.md, server/restore_data.py, server/preflight.py and
server/train.py. Single front RGB plus 13 joint positions -> five 13-joint targets.
The frozen visual/language model uses one image, without fabricated wrist images
or an 8D Cartesian state. The new readout is independent of the legacy 7D decoder.

The environment in environments/so101_wrist_side and the old scripts/train_limited.py,
scripts/evaluate_limited.py, scripts/validate_wide_dataset.py and configs/so101_vla.json
are legacy Cartesian experiments. They must not be used to evaluate this dataset.
The authoritative new physics assets are supplied inside the dataset Release.
No new closed-loop rollout or hardware success claim is made.
