"""Explicit entrypoint for issued-command-conditioned Policy training."""
import subprocess,sys
from pathlib import Path
if __name__=='__main__':
 command=[sys.executable,'-u',str(Path(__file__).with_name('train_bridge.py')),'--command-context',*sys.argv[1:]]
 raise SystemExit(subprocess.run(command).returncode)
