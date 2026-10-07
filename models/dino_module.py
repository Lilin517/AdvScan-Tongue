import copy
import torch
import torch.nn as nn
from .backbone import BackboneWithMyBlock
from .dino_head import DINOHead

class DINOModule(nn.Module):

    def __init__(self, backbone_name: str, stage, version: str, cfg, pretrain_ckpt_path: str=None):
        super().__init__()
        self.backbone_name = backbone_name
        self.stage = stage
        self.version = version
        self.student_backbone = BackboneWithMyBlock(backbone_name, stage=stage, pretrained=True, version=version, pretrain_ckpt_path=pretrain_ckpt_path)
        self.student_head = DINOHead(in_dim=self.student_backbone.feature_dim, hidden_dim=cfg.get('dino_projector_hidden_dim', 2048), bottleneck_dim=cfg.get('dino_projector_bottleneck_dim', 256), n_prototypes=cfg.get('dino_n_prototypes', 65536), norm_last_layer=True)
        self.teacher_backbone = BackboneWithMyBlock(backbone_name, stage=stage, pretrained=True, version=version, pretrain_ckpt_path=pretrain_ckpt_path)
        self.teacher_head = DINOHead(in_dim=self.teacher_backbone.feature_dim, hidden_dim=cfg.get('dino_projector_hidden_dim', 2048), bottleneck_dim=cfg.get('dino_projector_bottleneck_dim', 256), n_prototypes=cfg.get('dino_n_prototypes', 65536), norm_last_layer=True)
        for student_p, teacher_p in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            teacher_p.data.copy_(student_p.data)
            teacher_p.requires_grad = False
        for student_p, teacher_p in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher_p.data.copy_(student_p.data)
            teacher_p.requires_grad = False
        self.current_momentum = cfg.get('dino_teacher_momentum', 0.996)

    @torch.no_grad()
    def _momentum_update_teacher(self, momentum: float=None):
        if momentum is None:
            momentum = self.current_momentum
        for student_p, teacher_p in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            teacher_p.data.mul_(momentum).add_(student_p.data, alpha=1 - momentum)
        for student_p, teacher_p in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher_p.data.mul_(momentum).add_(student_p.data, alpha=1 - momentum)

    def forward_student(self, x):
        features = self.student_backbone(x)
        if isinstance(features, dict):
            features = features['cls']
        logits, bottleneck = self.student_head(features)
        return (logits, bottleneck)

    @torch.no_grad()
    def forward_teacher(self, x):
        features = self.teacher_backbone(x)
        if isinstance(features, dict):
            features = features['cls']
        logits, bottleneck = self.teacher_head(features)
        return (logits, bottleneck)

    def forward(self, global_crops, local_crops=None):
        n_global = len(global_crops)
        teacher_cat = torch.cat(global_crops, dim=0)
        teacher_logits_cat, _ = self.forward_teacher(teacher_cat)
        teacher_logits = list(torch.chunk(teacher_logits_cat, n_global, dim=0))
        if local_crops is not None:
            all_crops = torch.cat(global_crops + list(local_crops), dim=0)
            n_views = n_global + len(local_crops)
        else:
            all_crops = teacher_cat
            n_views = n_global
        student_logits_cat, _ = self.forward_student(all_crops)
        student_logits = list(torch.chunk(student_logits_cat, n_views, dim=0))
        return (teacher_logits, student_logits)
