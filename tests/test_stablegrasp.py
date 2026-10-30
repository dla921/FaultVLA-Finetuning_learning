import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from faultvla.stablegrasp import split_ids,samples,statistics
class NativeDataTests(unittest.TestCase):
 def test_alignment_and_train_only_statistics(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);ids=[f'episode_{i:03d}' for i in range(200)]
   m=dict(train=ids[:160],validation=ids[160:180],test=ids[180:])
   (r/'manifest_200.json').write_text(json.dumps(dict(splits=m)))
   for i,name in enumerate(ids[:161]):
    ep=r/'episodes'/name;ep.mkdir(parents=True)
    x=np.arange(130).reshape(10,13).astype(float)
    if i==160:x[:]=1e9
    np.savez(ep/'data.npz',qpos=x,action=x+1,rgb_state_index=[0,5])
   ep=r/'episodes'/ids[0];(ep/'rgb').mkdir()
   (ep/'metadata.json').write_text(json.dumps(dict(accepted=True,images_saved=True,task_success=True,instruction='transfer')))
   for k in [0,5]:Image.new('RGB',(224,224)).save(ep/'rgb'/f'{k:06d}.png')
   rows=list(samples(r,'train',1,2));self.assertEqual(len(rows),2)
   np.testing.assert_array_equal(rows[1]['action'],np.arange(130).reshape(10,13)[5:10]+1)
   np.testing.assert_array_equal(rows[1]['state'],np.arange(130).reshape(10,13)[5])
   stat=statistics(r)
   np.testing.assert_allclose(stat['qpos']['mean'],np.arange(130).reshape(10,13).mean(0))
   m['test'][0]=m['train'][0];(r/'manifest_200.json').write_text(json.dumps(dict(splits=m)))
   with self.assertRaises(ValueError):split_ids(r,'train')
if __name__=='__main__':unittest.main()
