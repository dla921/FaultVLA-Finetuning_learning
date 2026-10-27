import argparse,numpy as np
from PIL import Image
from control import Robot,ROOT,save_result
p=argparse.ArgumentParser();p.add_argument('--camera',default='both',choices=['both','front_rgbd','overhead_rgbd']);a=p.parse_args();r=Robot();r.step(.2);out={}
for n in (['front_rgbd','overhead_rgbd'] if a.camera=='both' else [a.camera]):
 rgb,depth=r.rgbd(n);Image.fromarray(rgb).save(ROOT/'results'/f'{n}.png');np.save(ROOT/'results'/f'{n}_depth_m.npy',depth)
 out[n]={'rgb_shape':list(rgb.shape),'depth_shape':list(depth.shape),'finite':bool(np.isfinite(depth).all()),'depth_range_m':[float(depth.min()),float(depth.max())]}
save_result('cameras.json',out)
