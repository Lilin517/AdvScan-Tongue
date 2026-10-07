import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOHead(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int=2048, bottleneck_dim: int=256, n_prototypes: int=65536, norm_last_layer: bool=True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim)])
        self.mlp = nn.Sequential(*layers)
        self.prototypes = nn.utils.weight_norm(nn.Linear(bottleneck_dim, n_prototypes, bias=False))
        self.prototypes.weight_g.data.fill_(1.0)
        if norm_last_layer:
            self.prototypes.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        bottleneck = x
        logits = self.prototypes(x)
        return (logits, bottleneck)
