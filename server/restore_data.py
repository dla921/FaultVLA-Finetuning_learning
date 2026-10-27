"""Restore release data with checksums and confined, regular-file-only extraction."""
import argparse,hashlib,json,subprocess,tarfile,os
from pathlib import Path,PurePosixPath
R=Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--local-assets',type=Path);a=p.parse_args()
 m=json.loads((R/'configs/data_release.json').read_text());cache=a.local_assets or R/'dist'
 cache.mkdir(parents=True,exist_ok=True)
 for asset in m['assets']:
  target=cache/asset['name']
  if not target.exists():
   if a.local_assets:raise FileNotFoundError(target)
   subprocess.run(['gh','release','download',m['tag'],'--repo',m['repository'],'--pattern',asset['name'],'--dir',str(cache)],check=True)
  if target.stat().st_size!=asset['size'] or digest(target)!=asset['sha256']:raise ValueError('Corrupt archive: '+str(target))
  print('Verified',target.name,flush=True)
 destination=R/'data';destination.mkdir(exist_ok=True)
 base=destination/m['dataset'];marker=base/'.release-restored.json'
 if marker.exists() and json.loads(marker.read_text())==m:
  print('Already restored; run server/preflight.py to recheck data');return
 stage=destination/(m['dataset']+'.extracting')
 stage.mkdir(exist_ok=True)
 # A failed extraction may be retried in this dedicated staging directory.
 for asset in m['assets']:
  with tarfile.open(cache/asset['name'],'r:gz') as tar:
   for member in tar:
    path=PurePosixPath(member.name)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!=m['dataset'] or not member.isfile():
     raise ValueError('Unsafe archive member: '+member.name)
    out=(stage.joinpath(*path.parts[1:])).resolve()
    if not out.is_relative_to(stage.resolve()):raise ValueError('Path escaped extraction directory')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,out.open('wb') as dst:
     import shutil
     shutil.copyfileobj(src,dst,2**20)
  print('Extracted',asset['name'],flush=True)
 if base.exists():raise FileExistsError('Existing dataset preserved: '+str(base))
 (stage/'.release-restored.json').write_text(json.dumps(m,indent=2));stage.rename(base)
 print('Restored',base)
if __name__=='__main__':main()
