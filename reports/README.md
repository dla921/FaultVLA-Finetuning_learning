# Validation scope

- stablegrasp200_validation.json: full source dataset integrity check.
- separated_smoke_training.json: current RGB-only separated arm/hand/vision GPU check.
- unit_tests_separated.txt: current complete unit-test run.
- stablegrasp_smoke_training.json and unit_tests_stablegrasp.txt: historical 13-axis implementation.
- Other reports: previous wide200/Cartesian package, not current validation.

No full training, closed-loop success, real hardware or fault-recovery performance
has been established by a smoke run. Legacy artifacts are preserved pending approval.

- mujoco_evaluation_validation.json: original-action physics replay and short RGB-policy connection only; not a trained baseline task-success result.

- command_limiter_validation.json and unit_tests_command_limiter.txt: optional command-rate filter bounds and short simulation checks, not task success.

- bridge_policy_validation.json and unit_tests_bridge_policy.txt: actual Pro Policy GPU weight updates, old/new offline diagnostic and short MuJoCo reload checks. No full model task-success claim.

- grasp_diagnostics_validation.json and unit_tests_grasp_diagnostics.txt: read-only actual joint/palm/cup telemetry and same-episode comparison; no control changes.

- visual_hand_validation.json and unit_tests_visual_hand.txt: optional camera-stage hand supervisor smoke/logic checks; not a full-trained grasp-success result.

- rollout_start_validation.json and unit_tests_rollout_start.txt: expert-prefix diagnostic initialization, source-index continuity and explicit exclusion from full-task success.
