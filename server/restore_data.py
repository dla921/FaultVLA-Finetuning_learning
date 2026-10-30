"""Download public ZIP assets and restore with checksums and confined extraction."""
import argparse,hashlib,json,shutil,urllib.request,zipfile
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
   url=f"https://github.com/{m['repository']}/releases/download/{m['tag']}/{asset['name']}"
   partial=target.with_suffix('.partial')
   with urllib.request.urlopen(url) as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst,2**20)
   partial.rename(target)
  if target.stat().st_size!=asset['size'] or digest(target)!=asset['sha256']:raise ValueError('Corrupt archive: '+str(target))
  print('Verified',target.name,flush=True)
 destination=R/'data';destination.mkdir(exist_ok=True)
 base=destination/m['dataset'];marker=base/'.release-restored.json'
 if marker.exists() and json.loads(marker.read_text())==m:
  print('Already restored; run server/preflight.py --data-only');return
 if base.exists():raise FileExistsError('Existing dataset preserved: '+str(base))
 stage=destination/(m['dataset']+'.extracting');stage.mkdir(exist_ok=True)
 seen=set()
 for asset in m['assets']:
  with zipfile.ZipFile(cache/asset['name']) as z:
   for member in z.infolist():
    path=PurePosixPath(member.filename)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!=m['dataset'] or '\\' in member.filename:
     raise ValueError('Unsafe member: '+member.filename)
    if member.is_dir():continue
    if member.filename in seen or (member.external_attr>>16)&0o170000==0o120000:raise ValueError('Duplicate/symlink')
    seen.add(member.filename)
    out=stage.joinpath(*path.parts[1:]).resolve()
    if not out.is_relative_to(stage.resolve()):raise ValueError('Path escaped stage')
    out.parent.mkdir(parents=True,exist_ok=True)
    with z.open(member) as src,out.open('wb') as dst:shutil.copyfileobj(src,dst,2**20)
  print('Extracted',asset['name'],flush=True)
 (stage/'.release-restored.json').write_text(json.dumps(m,indent=2));stage.rename(base)
 print('Restored',base)
if __name__=='__main__':main()
