"""Run the untouched demo in a new copy so supplied outputs remain intact."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--viewer', action='store_true')
p.add_argument('--video', action='store_true')
a = p.parse_args()
out = root/'results'/('baseline_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
shutil.copytree(root/'environments/so101_wrist_side', out/'demo')
cmd = [sys.executable, 'side_transfer.py']
if a.viewer: cmd.append('--viewer')
if a.video: cmd.append('--video')
print('Results:', out/'demo/results', flush=True)
raise SystemExit(subprocess.call(cmd, cwd=out/'demo'))
