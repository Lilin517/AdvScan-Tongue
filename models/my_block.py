import torch
import torch.nn as nn

class _MyBlockBase(nn.Module):

    def _zero_init_all(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.zeros_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

class MyBlock_V2_3(_MyBlockBase):

    def __init__(self, dim: int, expansion: int=4):
        super().__init__()
        self.norm_1 = LayerNorm2d(dim)
        self.norm_2 = LayerNorm2d(2 * dim)
        self.act = nn.GELU()
        self.conv_15 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 5), padding=(0, 2))
        self.conv_51 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(5, 1), padding=(2, 0))
        self.conv_11 = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=(1, 1))
        self._zero_init_all()

    def forward(self, x):
        save_x = x
        x = self.norm_1(x)
        x_1 = self.conv_15(x)
        x_2 = self.conv_51(x)
        x = torch.cat([x_1, x_2], dim=1)
        x = self.act(x)
        x = self.norm_2(x)
        x = self.conv_11(x)
        x = self.act(x)
        x = x + save_x
        return x

class LayerNorm2d(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

def create_my_block(dim, version="v2_3"):
    if version.lower() != "v2_3":
        raise ValueError("Only the best-checkpoint v2_3 block is included")
    return MyBlock_V2_3(dim)
