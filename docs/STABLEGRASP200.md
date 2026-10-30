# Camera-only separated control

Current workflow: README.md and server/{restore_data,preflight,train}.py.
Input: front RGB and task text. Frozen VLM -> trained arm5 / hand2 / visual3 readout.
Arm labels are the first five original action channels. HandController.encode
validates and converts the eight hand channels to two independent synergies;
HandController.targets reconstructs them without inventing measured sensor state.
No hand/arm actual joint positions are fed into this policy.

Image t supervises five consecutive commands [t,t+100ms) at 50 Hz. Visual location
labels at t are generated offline from cup_xyz, with a 15mm elevation threshold
above both initial/final rest heights. This uses ground truth only for supervision;
policy prediction accepts no simulator state. The monitor's location probabilities
are not evidence of contact force, task success or a specific failed actuator.

Inference: SeparatedPolicy.predict(rgb,task) returns synchronized arm and hand
chunks plus visual probabilities. It does not select recovery actions. scripts/evaluate_separated.py implements
a simulated executor with 50 Hz commands and 10 Hz camera replanning. Simulation
time pauses during inference; real-time hardware control is not validated. A temporal failure monitor and recovery demonstrations remain
future work. Normal-only data cannot validate fault diagnosis or fault recovery.

Old joint_readout.pt files must be retrained as separated_policy.pt. Previous
Cartesian adapters, wrist-camera environment and train_limited/evaluate_limited
scripts remain legacy references and are not the current inference path.
