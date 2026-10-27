"""Download a pinned upstream Pro checkpoint; do not run remote Python code."""
import argparse,json
from pathlib import Path
from huggingface_hub import snapshot_download
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--model',choices=['object','spatial'],default='object');a=p.parse_args()
c=json.loads((R/'configs/checkpoints.json').read_text())[a.model]
path=snapshot_download(repo_id=c['checkpoint_repo'],revision=c['checkpoint_revision'],local_dir=R/c['checkpoint'],ignore_patterns=['*.log'])
print(path)
