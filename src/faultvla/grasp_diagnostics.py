"""Read-only simulator telemetry for evaluation, never a policy/controller input.
Distances describe geometry; proximity alone does not prove a feasible grasp.
"""
import numpy as np

def observe(model,data):
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
    vadr=model.jnt_dofadr[model.actuator_trnid[:,0]]
    cup=data.body('demo_cube');palm=data.site('palm_target')
    hand_rotation=data.body('r_wrist_interface').xmat.reshape(3,3)
    delta=cup.xpos-palm.xpos
    contacts=[];hand_contacts=[];cup_id=model.body('demo_cube').id
    for contact in data.contact:
        first=model.geom_bodyid[contact.geom1]==cup_id
        second=model.geom_bodyid[contact.geom2]==cup_id
        if first==second:continue
        other=contact.geom2 if first else contact.geom1
        name=model.geom(other).name or ''
        contacts.append(name)
        if name.startswith('contact_'):hand_contacts.append(name)
    return dict(actual_joint_positions_rad=data.qpos[qadr].tolist(),
                actual_joint_velocities_rad_s=data.qvel[vadr].tolist(),
                target_tracking_error_rad=(data.ctrl-data.qpos[qadr]).tolist(),
                palm_site_world_m=palm.xpos.tolist(),cup_minus_palm_world_m=delta.tolist(),
                cup_minus_palm_hand_frame_m=(hand_rotation.T@delta).tolist(),
                cup_rotation_in_hand_frame=(hand_rotation.T@cup.xmat.reshape(3,3)).tolist(),
                palm_cup_center_distance_m=float(np.linalg.norm(delta)),
                cup_contacts=contacts,cup_hand_contacts=hand_contacts,
                note='Simulator ground truth for diagnosis only; palm_target is a reference site, not a grasp-success sensor')

class DiagnosticSummary:
    def __init__(self):
        self.closest=None;self.final=None;self.contact_steps=0
        self.max_tracking=np.zeros(13)
    def update(self,time,record):
        entry=dict(time=float(time),**record)
        if self.closest is None or record['palm_cup_center_distance_m']<self.closest['palm_cup_center_distance_m']:
            self.closest=entry
        self.final=entry;self.contact_steps+=bool(record['cup_hand_contacts'])
        self.max_tracking=np.maximum(self.max_tracking,np.abs(record['target_tracking_error_rad']))
    def result(self):
        return dict(closest_approach=self.closest,final=self.final,
                    steps_with_hand_cup_contact=self.contact_steps,
                    max_abs_target_tracking_error_rad=self.max_tracking.tolist(),
                    scope='Evaluation-only geometry/contact truth; not exposed to either learned policy or visual hand controller. No alignment/success inferred from distance alone.')
