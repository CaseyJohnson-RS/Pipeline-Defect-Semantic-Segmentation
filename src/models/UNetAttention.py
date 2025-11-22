import torch.nn as nn
from segmentation_models_pytorch import Unet


class UNetAttention(Unet):
    """
    smp.Unet + a built-in attention gate and dropout ONLY in the bottleneck.
    No changes to the encoder/decoder/head architecture.
    """
    def __init__(self, dropout_rate=0.2, **unet_kw):
        super().__init__(**unet_kw)

        c = self.encoder.out_channels[-1]          # channels of the last feature map
        self.att = nn.Sequential(
            nn.Conv2d(c, c // 2, 1), nn.ReLU(),
            nn.Conv2d(c // 2, 1, 1), nn.Sigmoid()
        )
        self.drop = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        features = self.encoder(x)
        features[-1] = self.drop(features[-1] * self.att(features[-1]))
        return self.segmentation_head(self.decoder(features))