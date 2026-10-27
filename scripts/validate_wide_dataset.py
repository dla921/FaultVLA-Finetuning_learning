"""Audit the completed wide-table dataset and all included RGB files."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from faultvla.limited_training import load_split,training_statistics
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/cartesian_demos_v2_wide200');a=p.parse_args()
base=a.data;m=json.loads((base/'manifest.json').read_text())
assert m['complete'], 'Collection not complete'
assert len(m['episodes'])==200
assert len({e['id'] for e in m['episodes']})==200
assert len({tuple(e['layout']['cup_xyz']+e['layout']['goal_xy']) for e in m['episodes']})==200
accepted={e['id'] for e in m['episodes']};assert not accepted.intersection(e['id'] for e in m['excluded'])
report=dict(validated=False,episodes=0,observations=0,images=0,splits={},excluded_attempts=len(m['excluded']))
gaps=[];lifts=[];settles=[]
for number,e in enumerate(m['episodes']):
 ep=base/e['path'];meta=json.loads((ep/'metadata.json').read_text());quality=json.loads((ep/'quality.json').read_text())
 assert meta['success'] and meta['error'] is None and not meta['rejected'] and not meta['clipped']
 assert meta['layout']==e['layout'] and e['split']==meta['layout']['split']
 assert meta['mode']=='expert_target_feedback_via_7d'
 profile=json.loads((ep/'rendering.json').read_text())
 assert profile['backend']=='glfw/d3d12' and 'NVIDIA' in profile['renderer']
 assert not profile['shadows'] and not profile['reflections'] and profile['multisamples']==0
 assert quality['validated'] and not quality['fault_injection'] and quality['action_adapter_used']
 n=e['records'];assert n==meta['records']==quality['records']==470
 with np.load(ep/'trajectory.npz') as t:
  assert t['actions_7d'].shape==(n,7) and t['state_8d'].shape==(n,8)
  for k in t.files:assert len(t[k])==n and np.isfinite(t[k]).all(),(e['id'],k)
  assert np.max(np.abs(t['actions_7d'][:,:6]))<=1+1e-8
  assert set(np.unique(t['actions_7d'][:,6])).issubset({0.,1.})
  np.testing.assert_allclose(np.diff(t['time']),.05,atol=1e-8)
  rows=[json.loads(x) for x in (ep/'observations.jsonl').read_text().splitlines()];assert len(rows)==n
  for i,row in enumerate(rows):
   assert row['index']==i and row['time']==t['time'][i]
   assert row['is_first']==(i==0) and row['is_last']==(i==n-1)
   assert row['task']==meta['config']['task']
   np.testing.assert_array_equal(row['action'],t['actions_7d'][i]);np.testing.assert_array_equal(row['state'],t['state_8d'][i])
   assert set(row['images'])=={'front','wrist'}
   for camera,path in row['images'].items():
    assert path==f'rgb/{i:05d}_{camera}.png'
    with Image.open(ep/path) as image:assert image.size==(224,224) and image.mode=='RGB';image.verify()
    report['images']+=1
 gap=meta['layout']['table2_xy'][1]+.08-.10
 assert .110-1e-8<=gap<=.140+1e-8
 assert abs(gap-quality['table_edge_gap_y_m'])<1e-10
 assert quality['final_half_second_cup_displacement_m']<.001
 gaps.append(gap);lifts.append(meta['lift_m']);settles.append(quality['final_half_second_cup_displacement_m'])
 report['episodes']+=1;report['observations']+=n
 report['splits'][e['split']]=report['splits'].get(e['split'],0)+1
 if (number+1)%20==0:print(f'Validated {number+1}/200',flush=True)
for split in ['train','validation','test']:
 entries=load_split(base,split)
 assert entries==[e for e in m['episodes'] if e['split']==split]
assert report['splits']=={'train':160,'validation':20,'test':20}
# Exercise the actual limited-training loader; validation/test are not read for these statistics.
stats,ids=training_statistics(base)
assert len(ids)==160 and set(ids)=={e['id'] for e in m['episodes'] if e['split']=='train'}
report.update(validated=True,table_edge_gap_m=[min(gaps),max(gaps)],old_table_edge_gap_m=.06,
    lift_range_m=[min(lifts),max(lifts)],max_final_half_second_cup_displacement_m=max(settles),
    limited_training_loader_compatible=True,train_statistics_episode_count=len(ids),
    test_used_for_training=False)
(base/'validation_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

