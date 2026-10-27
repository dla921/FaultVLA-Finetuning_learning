import os,sys,tempfile
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from faultvla.bridge import Environment
import mujoco
from OpenGL import GL
with tempfile.TemporaryDirectory() as tmp:
 r=Environment(tmp)
 with mujoco.Renderer(r.model,224,224) as renderer:
  renderer.update_scene(r.data,camera='front_rgbd');renderer.render()
  print('GL_VENDOR',GL.glGetString(GL.GL_VENDOR));print('GL_RENDERER',GL.glGetString(GL.GL_RENDERER))

