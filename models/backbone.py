import os
import sys
import math
import torch
import torch.nn as nn
import timm
from .registry import get_backbone_info
from .my_block import create_my_block

class StageWithMyBlock(nn.Module):

    def __init__(self, base_stage: nn.Module, my_block: nn.Module, feature_format: str='bchw', grid_size: tuple=None):
        super().__init__()
        self.base_stage = base_stage
        self.my_block = my_block
        self.feature_format = feature_format
        self.grid_size = grid_size

    def forward(self, x):
        x = self.base_stage(x)
        if self.feature_format == 'bchw':
            x = x + self.my_block(x)
        elif self.feature_format == 'bhwc':
            x_bchw = x.permute(0, 3, 1, 2).contiguous()
            x_bchw = x_bchw + self.my_block(x_bchw)
            x = x_bchw.permute(0, 2, 3, 1).contiguous()
        elif self.feature_format == 'blc':
            cls_token = x[:, :1]
            patches = x[:, 1:]
            B, N, D = patches.shape
            if self.grid_size is not None:
                H, W = self.grid_size
            else:
                H = W = int(N ** 0.5)
            patches_bchw = patches.transpose(1, 2).reshape(B, D, H, W).contiguous()
            patches_bchw = patches_bchw + self.my_block(patches_bchw)
            patches = patches_bchw.reshape(B, D, N).transpose(1, 2).contiguous()
            x = torch.cat([cls_token, patches], dim=1)
        return x

def _split_vit(model):
    blocks = list(model.blocks.children())
    parts = [('patch_embed', _PatchEmbedWrapper(model.patch_embed, getattr(model, 'cls_token', None), getattr(model, 'pos_embed', None)))]
    for i, block in enumerate(blocks):
        parts.append((f'block_{i}', block))
    head = model.norm if hasattr(model, 'norm') else nn.Identity()
    return (parts, head)

class _PatchEmbedWrapper(nn.Module):

    def __init__(self, patch_embed, cls_token, pos_embed):
        super().__init__()
        self.patch_embed = patch_embed
        self.cls_token = cls_token
        self.pos_embed = pos_embed

    def forward(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        return x

class BackboneWithMyBlock(nn.Module):

    def __init__(self, backbone_name: str, stage, pretrained: bool=True, version: str='v1', pretrain_ckpt_path: str=None):
        assert backbone_name == 'ViT-Tiny' and version.lower() == 'v2_3' and (list(stage) == list(range(1, 12)))
        super().__init__()
        self.backbone_name = backbone_name
        info = get_backbone_info(backbone_name)
        self.family = info['family']
        self.feature_dim = info['feature_dim']
        self.input_size = info['input_size']
        if isinstance(stage, (list, tuple, set)):
            target_indices = sorted(set((int(s) for s in stage)))
        else:
            target_indices = [int(stage)]
        self.stage = target_indices
        self._target_indices = set(target_indices)
        self._info = info
        _use_imagenet = pretrained and (not (pretrain_ckpt_path is not None and os.path.isfile(pretrain_ckpt_path)))
        model_kwargs = {}
        if info['family'] == 'vit' and 'swin' not in info['timm_name'].lower():
            model_kwargs['img_size'] = info['input_size']
        self.model = timm.create_model(info['timm_name'], pretrained=_use_imagenet, num_classes=0, **model_kwargs)
        if pretrain_ckpt_path is not None and os.path.isfile(pretrain_ckpt_path):
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if _project_root not in sys.path:
                sys.path.insert(0, _project_root)
            ckpt = torch.load(pretrain_ckpt_path, map_location='cpu', weights_only=False)
            backbone_sd = ckpt.get('backbone_state_dict', ckpt)
            model_sd = self.model.state_dict()
            matched = {}
            for k, v in backbone_sd.items():
                clean_k = k
                if clean_k.startswith('module.'):
                    clean_k = clean_k[len('module.'):]
                if clean_k.startswith('student_backbone.'):
                    clean_k = clean_k[len('student_backbone.'):]
                if clean_k.startswith('model.'):
                    clean_k = clean_k[len('model.'):]
                if clean_k in model_sd and v.shape == model_sd[clean_k].shape:
                    matched[clean_k] = v
            if matched:
                self.model.load_state_dict(matched, strict=False)
        timm_name_lower = info['timm_name'].lower()
        self._parts, self._head = _split_vit(self.model)
        self._backbone_type = 'vit'
        if self._backbone_type == 'swin':
            feature_format = 'bhwc'
            grid_size = None
        elif self._backbone_type == 'vit':
            feature_format = 'blc'
            pe = self.model.patch_embed
            if hasattr(pe, 'patch_size'):
                ps = pe.patch_size
                ph = ps[0] if isinstance(ps, tuple) else ps
                pw = ps[1] if isinstance(ps, tuple) else ps
                grid_size = (self.input_size // ph, self.input_size // pw)
            else:
                grid_size = None
        else:
            feature_format = 'bchw'
            grid_size = None
        self.my_blocks = nn.ModuleList()
        for target_idx in target_indices:
            dim = self._detect_dim_at(target_idx)
            if dim is None:
                raise RuntimeError(f'Could not determine feature dimension at stage {target_idx} for {backbone_name}.')
            my_block = create_my_block(dim, version)
            target_name, target_module = self._parts[target_idx]
            self._parts[target_idx] = (target_name, StageWithMyBlock(target_module, my_block, feature_format=feature_format, grid_size=grid_size))
            self.my_blocks.append(my_block)
        self._feature_format = feature_format
        self.stages = nn.ModuleList([mod for _, mod in self._parts])

    @property
    def my_block(self):
        return self.my_blocks[0] if self.my_blocks else None

    def _detect_dim_at(self, target_idx: int):
        device = next(self.model.parameters()).device
        dummy = torch.randn(1, 3, self.input_size, self.input_size).to(device)
        with torch.no_grad():
            x = dummy
            for i in range(target_idx):
                _, part = self._parts[i]
                x = part(x)
            target_module = self._parts[target_idx][1]
            x = target_module(x)
            if x.dim() == 4:
                if self._backbone_type == 'swin':
                    return x.shape[-1]
                else:
                    return x.shape[1]
            elif x.dim() == 3:
                return x.shape[-1]
            else:
                return x.shape[-1]

    def forward_features(self, x):
        for _, stage in self._parts:
            x = stage(x)
        return x

    def forward(self, x):
        for _, stage in self._parts:
            x = stage(x)
        if self._backbone_type in ('vit', 'swin'):
            if self._head is not None and (not isinstance(self._head, nn.Identity)):
                x = self._head(x)
            if x.dim() == 3:
                return {'cls': x[:, 0], 'patch_tokens': x[:, 1:], 'features': x}
            elif x.dim() == 4:
                cls_token = x.mean([1, 2])
                patch_tokens = x.flatten(1, 2)
                return {'cls': cls_token, 'patch_tokens': patch_tokens, 'features': x}
            else:
                return {'cls': x, 'patch_tokens': None, 'features': x}
        else:
            x = self._head(x)
            if x.dim() == 4:
                x = x.mean([-2, -1])
            elif x.dim() == 3:
                x = x.mean(1)
            if x.dim() > 2:
                x = x.flatten(1)
            return x

    def _summarize_module(self, mod: nn.Module) -> str:
        name = mod.__class__.__name__
        children = list(mod.children())
        if not children:
            n_params = sum((p.numel() for p in mod.parameters()))
            if n_params == 0:
                return name
            return f'{name}({n_params:,} params)'
        if name in ('Sequential', 'ModuleList'):
            child_names = [c.__class__.__name__ for c in children]
            from collections import Counter
            dominant = Counter(child_names).most_common(1)[0][0]
            if len(set(child_names)) == 1:
                return f'Sequential({len(children)}×{dominant})'
            else:
                return f'Sequential({len(children)} modules: {', '.join(set(child_names))})'
        n_params = sum((p.numel() for p in mod.parameters()))
        child_names = [c.__class__.__name__ for c in children]
        unique = list(dict.fromkeys(child_names))
        return f'{name}({len(children)} sub: {', '.join(unique[:4])}{('…' if len(unique) > 4 else '')}, {n_params:,} params)'

    def architecture_summary(self) -> str:
        lines = []
        w = 80
        total_mb_params = sum((p.numel() for p in self.my_blocks.parameters()))
        mb_names = [mb.__class__.__name__ for mb in self.my_blocks]
        unique_names = list(dict.fromkeys(mb_names))
        mb_name_str = ', '.join(unique_names)
        stages_str = ','.join((str(s) for s in self.stage))
        lines.append('═' * w)
        lines.append(f'Architecture: {self.backbone_name} + {mb_name_str} @ stage=[{stages_str}] ({len(self.my_blocks)} blocks)')
        lines.append(f'family={self.family}  feature_dim={self.feature_dim}  input_size={self.input_size}  MyBlock_total_params={total_mb_params:,}')
        lines.append('─' * w)
        sorted_targets = sorted(self._target_indices)
        n_parts = len(self._parts)
        for i, (name, mod) in enumerate(self._parts):
            marker = '  ◀── 改造位置 (My_Block 在此插入)' if i in self._target_indices else ''
            summary = self._summarize_module(mod)
            if self._backbone_type == 'vit' and n_parts > 8 and (i not in self._target_indices):
                if i == 0:
                    pass
                elif i == n_parts - 1:
                    pass
                elif i == 1:
                    lines.append(f'  [{1:2d}..{n_parts - 2:2d}] {'block_*':16s} ({n_parts - 2} blocks, {len(self.my_blocks)} with MyBlock)')
                    continue
                else:
                    continue
            lines.append(f'  [{i:2d}] {name:16s} {summary}{marker}')
            if i in self._target_indices:
                mb_idx = sorted_targets.index(i)
                mb = self.my_blocks[mb_idx]
                mb_params = sum((p.numel() for p in mb.parameters()))
                base_summary = self._summarize_module(mod.base_stage)
                lines.append(f'        ├─ base_stage: {base_summary}')
                lines.append(f'        └─ my_block:  {mb.__class__.__name__} ({mb_params:,} params, zero-init, residual: x + MyBlock(x))')
        head_summary = self._summarize_module(self._head)
        lines.append(f'  {'head':16s} {head_summary}')
        lines.append('─' * w)
        return '\n'.join(lines)
