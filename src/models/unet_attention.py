import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetAttention(smp.Unet):
    """
    UNet с Attention Gate в bottleneck и Dropout для работы с плохими масками.
    Наследуется от smp.Unet для совместимости с сохранением/загрузкой.
    """
    
    def __init__(self, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        
        # Attention gate для bottleneck (подавляет фон, усиливает объект)
        bottleneck_channels = self.encoder.out_channels[-1]
        self.attention_gate = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
    def forward(self, x):
        """Forward pass with attention and dropout."""
        # Получаем features от encoder (список тензоров)
        features = self.encoder(x)
        
        # Применяем attention gate к самому глубокому feature map
        last_feature = features[-1]
        attention = self.attention_gate(last_feature)
        last_feature = last_feature * attention
        
        # Применяем dropout
        last_feature = self.dropout(last_feature)
        
        # Заменяем в списке features (ВАЖНО: не изменяем структуру списка)
        features[-1] = last_feature
        
        # === ИСПРАВЛЕНО: передаем features как ОДИН аргумент (список) ===
        decoder_output = self.decoder(features)
        
        # Финальная активация
        masks = self.segmentation_head(decoder_output)
        
        return masks