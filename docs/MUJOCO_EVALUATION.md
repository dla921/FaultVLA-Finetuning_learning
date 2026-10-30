# Current separated-policy MuJoCo evaluation

The original StableGrasp200 scene.xml is used for physics. scene_render.xml is
used only to render current physical qpos with the same front camera/LOD and
shadow/reflection flags as the source RGB. Original data files are never changed.
Without --episode, the first ID from the real test manifest is selected; numerical
episode order does not imply split membership.

## Environment/recorded-action physics check

```bash
python scripts/evaluate_separated.py --mode replay --video --output results/replay_check
```

This applies original recorded actions, not a learned model. It checks that source
commands and physical contacts can reproduce a transfer. The local tested version
is MuJoCo 3.17.0; data were collected with 3.3.7, so exact trajectory equality is
not claimed. --video writes rollout.gif without needing FFmpeg.

## Learned model closed-loop test

After training finishes:

```bash
python scripts/evaluate_separated.py --mode policy \
  --trained results/normal_baseline_01/separated_policy.pt \
  --video --output results/normal_baseline_01/mujoco_test_01
```

Only the initial recorded robot/object state and initial actuator velocities are
restored. Initial commands hold joint positions. Every 100ms a newly rendered RGB
image and the instruction are sent to SeparatedPolicy; five 20ms arm/hand commands
are executed. No recorded actions or later recorded states drive policy rollouts.
Simulator cup position/contact data are used only for evaluation/logging.
Simulation pauses during model inference: this does not demonstrate 10Hz wall-clock performance.

Outputs: summary.json, trace.jsonl, first.png, last.png and optional rollout.gif.
A full test runs the episode's original number of 50Hz steps. Task success requires
lift above 15mm, final XY within 30mm of the episode reference destination, upright
cosine > .95, final table2 contact and no named hand contact. This endpoint metric
does not establish a sustained hold or recovery performance. Full success exits 0;
full task failure exits 2; runtime errors exit 1. A --steps 10 connection check
sets task_success to null and is not a task-success evaluation.

Use a new output directory for every run. Repeat with each ID in manifest_200.json
splits.test to obtain the 20-episode test success rate. One episode is not a success rate.
For a visible window on a graphical machine use --backend glfw --viewer. On a
headless SSH server keep the default EGL and inspect the saved GIF/PNGs.
The local smoke checkpoint is not the user's remotely trained baseline.
