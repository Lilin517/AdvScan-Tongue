import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PRETRAIN_CKPT = os.path.join(_PROJECT_ROOT, 'output', 'pretrain', '正式全训练_GPU', 'ViT-Tiny', 'checkpoints', 'best.pth')
DEFAULT_CFG = {'epochs': 50, 'batch_size': 16, 'lr': 0.001, 'warmup_epochs': 5, 'weight_decay': 0.01, 'backbone_mode': 1, 'backbone_frozen': True, 'backbone_lr_scale': 0.1, 'mlp_hidden_dim': 512, 'mlp_dropout': 0.3, 'mode': 'mlp', 'grad_clip': 3.0, 'optimizer': 'adamw', 'amp': True, 'dataset_root': os.path.join(_PROJECT_ROOT, 'dataset', 'shezhenv3-coco-percent-3'), 'input_size': 224, 'num_workers': 8, 'prefetch_factor': 4, 'persistent_workers': True, 'pin_memory': True, 'print_freq': 10, 'val_freq': 5, 'phase': 'classifier', 'pretrain_dataset_root': os.path.join(_PROJECT_ROOT, 'dataset', 'mae_pretrain'), 'pretrain_epochs': 100, 'pretrain_batch_size': 8, 'pretrain_grad_accum_steps': 1, 'pretrain_lr': 0.00015, 'pretrain_wd': 0.05, 'dino_n_prototypes': 65536, 'dino_teacher_temp': 0.04, 'dino_student_temp': 0.1, 'dino_center_momentum': 0.9, 'dino_teacher_momentum': 0.996, 'dino_momentum_end': 1.0, 'dino_projector_hidden_dim': 2048, 'dino_projector_bottleneck_dim': 256, 'dino_local_crops_size': 96, 'dino_local_crops_number': 6, 'dino_global_crops_scale': [0.4, 1.0], 'dino_local_crops_scale': [0.05, 0.4], 'pretrain_ckpt_path': _DEFAULT_PRETRAIN_CKPT, 'gpus': 1, 'dist_backend': 'nccl', 'dist_url': 'env://'}

class Config:

    def __init__(self, defaults: dict=None):
        if defaults is None:
            defaults = DEFAULT_CFG
        self._data = dict(defaults)

    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        if name in self._data:
            return self._data[name]
        raise AttributeError(f'Config has no key: {name}')

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

    def update(self, overrides: dict):
        for k, v in overrides.items():
            if v is not None and k in self._data:
                self._data[k] = v

    def to_dict(self):
        return dict(self._data)

    def __repr__(self):
        return f'Config({self._data})'
DEFAULT_CFG.update(backbone_mode=2, backbone_frozen=False, mode='bce')
