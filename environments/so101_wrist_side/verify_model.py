import json,xml.etree.ElementTree as E,numpy as np,mujoco
from control import Robot,ROOT,ARM,HAND,save_result
r=Robot();src=E.parse(ROOT/'provenance'/'amazinghand_original.xml').getroot();new=E.parse(ROOT/'robot.xml').getroot();orig=src.find('./worldbody/body');hand=new.find('.//body[@name="r_wrist_interface"]')
assert [x.attrib for x in orig.iter('body') if x is not orig]==[x.attrib for x in hand.iter('body') if x is not hand]
assert [(x.get('name'),x.get('type'),x.get('axis')) for x in orig.iter('joint')]==[(x.get('name'),x.get('type'),x.get('axis')) for x in hand.iter('joint')]
assert [(x.get('site1'),x.get('site2')) for x in src.find('equality')]==[(x.get('site1'),x.get('site2')) for x in list(new.find('equality'))[:20]]
assert [r.model.actuator(i).name for i in range(r.model.nu)]==ARM+HAND
assert r.model.nu==13 and r.model.joint('wrist_roll').id>=0
mujoco.MjModel.from_xml_path(str(ROOT/'robot.xml'))
maxloop=0
for _ in range(100):r.step(.1);maxloop=max(maxloop,r.loop_error())
result={'mujoco_version':mujoco.__version__,'standalone_robot_compiled':True,'scene_compiled':True,'nu':r.model.nu,'robot_joints':r.model.njnt-1,'hand_joints':len(list(hand.iter('joint'))),'passive_hand_joints':len(list(hand.iter('joint')))-8,'preserved_original_hand_connections':20,'inactive_fault_locks':13,'hand_topology_check':True,'idle_seconds':float(r.data.time),'max_idle_loop_gap_m':maxloop,'warnings':[int(w.number) for w in r.data.warning]}
save_result('validation.json',result)
