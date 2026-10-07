import torch.nn as nn

class MLPHead(nn.Module):

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int=512, dropout: float=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        return self.net(x)
