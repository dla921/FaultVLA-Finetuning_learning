"""Compare evaluation-only geometry telemetry, without changing a policy."""
import argparse,json,math
from pathlib import Path

def read_run(path):
    summary=json.loads((path/'summary.json').read_text())
    rows=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
    if not rows or any('diagnostics' not in row for row in rows):
        raise ValueError('Rerun evaluation with current code to obtain diagnostics: '+str(path))
    return summary,rows

def describe(rows):
    closest=min(rows,key=lambda row:row['diagnostics']['palm_cup_center_distance_m'])
    d=closest['diagnostics']
    return dict(closest_time_s=closest['time'],closest_distance_mm=1000*d['palm_cup_center_distance_m'],
                cup_minus_palm_hand_frame_mm=[1000*x for x in d['cup_minus_palm_hand_frame_m']],
                actual_arm_at_closest_rad=d['actual_joint_positions_rad'][:5],
                requested_hand_targets_at_closest_rad=closest['command'][5:],
                steps_with_hand_cup_contact=sum(bool(row['diagnostics']['cup_hand_contacts']) for row in rows),
                max_arm_tracking_error_deg=max(abs(x) for row in rows for x in row['diagnostics']['target_tracking_error_rad'][:5])*180/math.pi)

def compare(policy_summary,policy_rows,replay_summary,replay_rows):
    if policy_summary['episode']!=replay_summary['episode']:raise ValueError('Use the same initial episode')
    if policy_summary['mode']!='policy' or replay_summary['mode']!='replay':raise ValueError('Expected policy and replay runs')
    lookup={row['step']:row for row in replay_rows}
    differences=[]
    for row in policy_rows:
        other=lookup.get(row['step'])
        if other is None:continue
        if abs(row['time']-other['time'])>1e-6:raise ValueError('Trace times differ')
        pd=row['diagnostics'];rd=other['diagnostics']
        differences.append(sum((x-y)**2 for x,y in zip(pd['palm_site_world_m'],rd['palm_site_world_m']))**.5)
    if not differences:raise ValueError('No matched steps')
    return dict(episode=policy_summary['episode'],policy=describe(policy_rows),replay=describe(replay_rows),
                policy_task_success=policy_summary['task_success'],replay_task_success=replay_summary['task_success'],
                matched_steps=len(differences),same_time_mean_palm_difference_mm=1000*sum(differences)/len(differences),
                note='Same-time trajectory differences can reflect different task timing. Closest distance alone does not establish alignment or grasp readiness. Simulator telemetry is evaluation-only.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy-run',type=Path,required=True)
    p.add_argument('--replay-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    ps,pr=read_run(a.policy_run);rs,rr=read_run(a.replay_run);result=compare(ps,pr,rs,rr)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
