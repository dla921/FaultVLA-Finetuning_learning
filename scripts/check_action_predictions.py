"""Offline teacher-image action errors; no physics, no weight updates."""
import argparse,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'VLA-Adapter')]
import numpy as np,torch
from faultvla.stablegrasp import separated_samples,VISUAL_STATES
from faultvla.stablegrasp_policy import load_policy
from faultvla.hand_controller import HandController

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trained',type=Path,required=True)
 p.add_argument('--split',choices=['train','validation','test'],default='validation')
 p.add_argument('--samples-per-episode',type=int,default=20);p.add_argument('--max-episodes',type=int)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.samples_per_episode<1 or (a.max_episodes is not None and a.max_episodes<1):p.error('Positive sample counts required')
 a.output.mkdir(parents=True,exist_ok=False)
 payload=torch.load(a.trained,map_location='cpu',weights_only=True)
 c=json.loads((R/'configs/checkpoints.json').read_text())[payload['base_model']]
 policy=load_policy(R/c['checkpoint'],c['unnorm_key'],a.trained)
 errors=[];raw_errors=[];hand_errors=[];visual_correct=0;saturated=0;per_episode={}
 with (a.output/'predictions.jsonl').open('w') as log:
  for sample in separated_samples(R/'data/SO101_StableGrasp200',a.split,a.max_episodes,a.samples_per_episode,evenly=True,command_statistics=payload['statistics'] if hasattr(policy,'set_offline_context') else None):
   if hasattr(policy,'set_offline_context'):policy.set_offline_context(sample['command_context'])
   result=policy.predict(sample['rgb'],sample['task'])
   applied=np.concatenate((result['arm_targets_rad'],result['hand_commands']),axis=1)
   raw=result['raw_commands'];target=sample['action'];error=applied-target
   errors.append(error);raw_errors.append(raw-target)
   hand_errors.append(result['hand_targets_rad']-HandController.targets(target[:,5:]))
   saturated+=int(np.any(np.abs(applied-raw)>1e-8))
   visual_correct+=int(result['visual_state']==VISUAL_STATES[sample['visual_label']])
   per_episode.setdefault(sample['episode'],[]).append(np.abs(error))
   log.write(json.dumps(dict(episode=sample['episode'],index=sample['index'],raw=raw.tolist(),applied=applied.tolist(),target=target.tolist(),visual_state=result['visual_state'],visual_label=VISUAL_STATES[sample['visual_label']]))+'\n')
   if len(errors)%20==0:print('Checked',len(errors),'frames',flush=True)
 if not errors:raise ValueError('No samples')
 e=np.abs(np.stack(errors));raw=np.abs(np.stack(raw_errors));hand=np.abs(np.stack(hand_errors))
 summary=dict(contract=payload['contract'],split=a.split,frames=len(errors),episodes=len(per_episode),
   arm_mae_rad=e[:,:,:5].mean((0,1)).tolist(),arm_mae_deg=np.rad2deg(e[:,:,:5].mean((0,1))).tolist(),
   arm_max_error_rad=e[:,:,:5].max((0,1)).tolist(),arm_mae_by_horizon_rad=e[:,:,:5].mean((0,2)).tolist(),
   raw_arm_mae_rad=raw[:,:,:5].mean((0,1)).tolist(),hand_command_mae=e[:,:,5:].mean((0,1)).tolist(),
   hand_motor_mae_rad=hand.mean((0,1)).tolist(),visual_accuracy=visual_correct/len(errors),
   frames_with_clipped_predictions=saturated,
   episode_mae={key:np.stack(value).mean((0,1)).tolist() for key,value in per_episode.items()},
   note='For context policies source past commands are teacher-forced; runtime uses own issued commands. Offline recorded-image errors only; no command-rate filter. Does not establish closed-loop success or a unique failure cause.')
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
