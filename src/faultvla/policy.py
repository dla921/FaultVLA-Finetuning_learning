"""Checkpoint-backed inference only: missing weights are a hard error."""
import json
import os
# Object/Spatial-Pro checkpoint dimensions must not depend on output filenames.
os.environ['VLA_ROBOT_PLATFORM'] = 'LIBERO'
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def validate_checkpoint(path, key):
    path = Path(path)
    required = ['config.json', 'dataset_statistics.json', 'preprocessor_config.json',
                'tokenizer_config.json', 'action_head--checkpoint.pt', 'proprio_projector--checkpoint.pt']
    missing = [f for f in required if not (path / f).is_file()]
    if not list(path.glob('*.safetensors')) and not (path / 'pytorch_model.bin').exists():
        missing.append('model weights (*.safetensors or pytorch_model.bin)')
    if missing:
        raise FileNotFoundError(f'Checkpoint incomplete: {path}\nMissing: {missing}\nRequired: complete VLA-Adapter Pro checkpoint; see README.md')
    stats = json.loads((path / 'dataset_statistics.json').read_text())
    if key not in stats:
        raise ValueError(f'Unknown statistics key {key}; available: {list(stats)}')
    for name, dim in [('action', 7), ('proprio', 8)]:
        for quantile in ['q01', 'q99']:
            values = np.asarray(stats[key][name][quantile])
            if values.shape != (dim,) or not np.isfinite(values).all():
                raise ValueError(f'Invalid {name}/{quantile} statistics')
    return stats


class Policy:
    def __init__(self, path, key, replan_actions=1):
        if not 1 <= replan_actions <= 8:
            raise ValueError('replan_actions must be in [1,8]')
        self.stats = validate_checkpoint(path, key)
        import torch
        import tensorflow as tf
        # TensorFlow is used only for image preprocessing; avoid GPU reservation.
        tf.config.set_visible_devices([], 'GPU')
        from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from experiments.robot.openvla_utils import get_action_head, get_proprio_projector, get_vla_action
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required for this BF16 Pro integration')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('This loader requires BF16 GPU support')
        AutoConfig.register('openvla', OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        self.cfg = SimpleNamespace(pretrained_checkpoint=str(Path(path).resolve()),
            use_l1_regression=True, use_pro_version=True, use_proprio=True,
            use_film=False, num_images_in_input=2, center_crop=True,
            num_open_loop_steps=replan_actions, unnorm_key=key, save_version='vla-adapter')
        # Direct local-class loading avoids upstream's checkpoint file rewriting.
        self.model = OpenVLAForActionPrediction.from_pretrained(
            self.cfg.pretrained_checkpoint, torch_dtype=torch.bfloat16,
            local_files_only=True, trust_remote_code=False, low_cpu_mem_usage=False).to('cuda:0').eval()
        self.model.set_version('vla-adapter')
        self.model.vision_backbone.set_num_images_in_input(2)
        self.model.norm_stats = self.stats
        self.head = get_action_head(self.cfg, self.model.llm_dim)
        self.proprio = get_proprio_projector(self.cfg, self.model.llm_dim, proprio_dim=8)
        self.processor = AutoProcessor.from_pretrained(self.cfg.pretrained_checkpoint,
                                                     trust_remote_code=False, local_files_only=True)
        self.infer = get_vla_action

    def predict(self, observation, task):
        if not task.strip():
            raise ValueError('Task instruction must not be empty')
        actions = np.asarray(self.infer(self.cfg, self.model, self.processor,
            {k: v.copy() for k, v in observation.items()}, task,
            action_head=self.head, proprio_projector=self.proprio,
            use_film=False, use_minivlm=True))
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f'Invalid policy output: {actions.shape}')
        return actions
