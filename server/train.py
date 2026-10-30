"""Run native StableGrasp200 training."""
import subprocess,sys
from pathlib import Path
if __name__=='__main__':
 subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/train_stablegrasp.py'),*sys.argv[1:]],check=True)
