import torch.nn as nn
from positional_encodings import torch_encodings as pos

from mutab.block import Blocks
from mutab.utils import MODELS


@MODELS.register_module()
class TabularEncoder(nn.Module):
    def __init__(self, d_model: int, backbone, **kwargs):
        super().__init__()

        self.backbone = backbone

        # blocks
        self.pos = pos.PositionalEncoding2D(d_model)
        self.enc = Blocks(d_model=d_model, **kwargs)

    def forward(self, img, train: bool, **kwargs):
        return dict(kwargs, img=self.process(self.backbone(img).permute(0, 2, 3, 1)))

    def process(self, img):
        assert img.ndim == 4

        # forward
        img = self.pos(img).add(img)
        hid = self.enc(x=img, y=img)

        return hid
