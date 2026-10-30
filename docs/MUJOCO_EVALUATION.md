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

## Optional command-rate diagnostic

The default has no rate limit and preserves the previous control behavior. To
compare against the existing unbounded episode_081 run, use the same checkpoint:

```bash
PYOPENGL_PLATFORM=egl python scripts/evaluate_separated.py --mode policy \
  --trained results/normal_baseline_01/separated_policy.pt --episode episode_081 \
  --arm-max-speed 1 --hand-max-speed 2 --video \
  --output results/normal_baseline_01/mujoco_rate_limited_01
```

These starting diagnostic values limit arm command changes to .02 rad per 20ms
and hand changes to .04 rad per 20ms. They are not tuned or validated physical
velocity limits: actual joint speeds can differ. The limiter remembers only the
previous command, initialized from the existing initial simulation state. It
uses no runtime measured joint state, object state, expert action or fault ID.
It runs for every 20ms command, including within chunks and at chunk boundaries.
Joint target bounds still apply. Either group can be limited independently.

trace.jsonl retains command as the applied command and adds requested_command
for the pre-rate-limit target (after joint clipping). summary.json records rates,
rate_limited_steps and per-actuator max_command_delta_rad, including the first
command relative to the initial hold command. With limits omitted, the summary
rates are null and behavior is unchanged.

Rate limiting may reduce violent command changes without fixing wrong actions.
Compare videos, lift and task_success; do not infer successful grasp from a lower
command delta. Diagnostic rate tuning on episode_081 makes it a development
case; reserve other episodes for final held-out reporting. In fault-tolerance
comparisons apply the same limiter settings to both algorithm-on and algorithm-off
conditions. Do not call this command filter a fault-tolerant algorithm.

## Evaluation-only grasp geometry diagnosis

Current runs add a diagnostics object to each post-step trace row and
summary.grasp_diagnostics. It records actual actuator joint positions/velocities,
command tracking error (target minus actual position), palm_target reference-site
world position, cup-minus-palm in world/hand coordinates, cup orientation in the
hand frame and named cup contacts. The hand frame is r_wrist_interface; positions
are metres, angles radians. These are simulator ground-truth readings solely for
post-run diagnosis, never inputs to the RGB policy, limiter or hand mapping.
This update changes no hand-control logic or learned weights.

The palm_target site is a model reference, not the skin surface. A small distance
alone is not proof of alignment: direction, orientation, finger placement and
contacts must also be inspected. Tracking error is not automatically a motor fault.

Run both modes again into NEW output folders; old traces lack these fields:

```bash
PYOPENGL_PLATFORM=egl python scripts/evaluate_separated.py --mode policy \
 --trained results/bridge_normal_01/bridge_policy.pt --episode episode_081 --video \
 --output results/bridge_normal_01/mujoco_alignment_01
PYOPENGL_PLATFORM=egl python scripts/evaluate_separated.py --mode replay \
 --episode episode_081 --video --output results/replay_alignment_01
python scripts/compare_grasp_diagnostics.py \
 --policy-run results/bridge_normal_01/mujoco_alignment_01 \
 --replay-run results/replay_alignment_01 \
 --output results/bridge_normal_01/alignment_comparison_01.json
```

These commands preserve the last unfiltered comparison (no rate-limit options).
The replay must match the policy episode. The compact comparison reports nearest
approach geometry, hand targets there, contact steps and actual tracking error.
Same-time differences can result from task timing differences; do not treat the
original trajectory as the only permissible policy trajectory. The policy run can
exit 2 for task failure while still writing all diagnostic outputs.

## Diagnostic start from an expert-aligned state

```bash
PYOPENGL_PLATFORM=egl python scripts/evaluate_separated.py --mode policy \
 --trained results/bridge_normal_01/bridge_policy.pt \
 --hand-judge results/hand_judge_01/hand_judge.pt \
 --episode episode_081 --start-phase 02_preclose_settle --video \
 --output results/bridge_normal_01/mujoco_aligned_start_01
```

This is not a full-task evaluation. --start-phase selects the first recorded
50Hz index in that phase. --start-index selects an explicit index instead; the
options are mutually exclusive. The default with neither option is unchanged.
The handoff phase 02_preclose_settle follows the original alignment movement and
precedes closure. Its index is read from this episode, not hardcoded.

To preserve passive/object velocities and contact dynamics, initialization replays
the original expert controls up to (but excluding) the selected index in the original
physics model. It does not teleport qpos alone. This reconstructs the expert state
in the currently installed MuJoCo version, not guaranteed bit-identical to the
3.3.7 recording. The prefix is excluded from the GIF, policy trace and diagnostic
rollout metrics. No policy or hand judge is running during the expert prefix.

At handoff, policy and hand supervisor start fresh. All subsequent policy-mode
controls come from current RGB, task text, the learned arm/hand outputs and optional
visual hand supervisor; source phase/time/pose is never fed to them. The first
camera-frame judge probabilities are reported as
visual_hand_supervisor.initial_probabilities as a class-name/probability mapping.
Observe whether initial readiness is high, whether it persists for three frames,
and whether close_command_triggered occurs. This does not force closure or hold
the arm fixed while waiting. High first-frame readiness that disappears can reflect
movement away from alignment rather than an unavailable close-command path.

Without --steps, run the remaining original horizon; --steps caps post-handoff
controls only. trace.step is local, trace.source_index is the reference timeline
index for logging only, trace.time is physics time including the prefix, and
elapsed_after_handoff_s is local elapsed simulation time. Events use physics time.
summary.initialization records the selected index/phase, method, prefix length and
actual initial diagnostics. complete_episode is false and task_success is null for
any explicit diagnostic start. completed_remaining_horizon and
diagnostic_remaining_task_success separately describe the remainder. Never count
this as a normal end-to-end test success. A completed remainder failure exits 2;
short connection checks exit 0 and leave remaining success null.

For a replay comparison add the identical --start-phase to mode=replay and omit
--trained/--hand-judge. The comparison tool rejects mismatched start indices.
A model failure here does not automatically imply the judge is wrong: inspect the
first-frame probabilities and subsequent geometry/commands together. No new
training, LoRA, threshold changes or independent arm-hold controller is introduced.
