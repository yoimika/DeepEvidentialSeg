import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

@runtime_checkable
class BaseConfig(Protocol):
    def make_model(self) -> nn.Module:
        ...


@dataclass(frozen=True)
class FPNFeatureExtractorConfig(BaseConfig):
    encoder_name: str = field(default="resnet50")
    encoder_weights: str = field(default="imagenet")
    in_channels: int = 3
    decoder_segmentation_channels: int = 128  # 输出特征的通道数

    def make_model(self, *, device='cpu') -> 'FPNFeatureExtractor':
        extractor = FPNFeatureExtractor(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=1,  # 此参数在提取特征时不再重要，但在初始化父类时是必须的
            decoder_segmentation_channels=self.decoder_segmentation_channels,
        ) 
        return extractor.to(device)

class FPNFeatureExtractor(smp.FPN):
    def forward(self, x):
        """
        重写 forward 方法以返回逐像素特征
        """
        self.check_input_shape(x)

        image_size = x.shape[-2:]

        # 1. 编码器 (Encoder) 提取多尺度特征
        features = self.encoder(x)

        # 2. 解码器 (Decoder) 融合特征
        # FPN Decoder 输出的是融合后的特征金字塔
        decoder_output = self.decoder(features)
        
        # 注意：此时 decoder_output 的通道数 = decoder_merge_policy 策略决定的通道数
        # 默认 'add' 策略下为 segmentation_channels (默认 128)
        # 如果你需要原始尺寸，可能还需要根据 output_stride 进行上采样

        output = nn.functional.interpolate(
            decoder_output,
            size=image_size,
            mode='bilinear',
            align_corners=False
        )
        
        return output
    









@dataclass(frozen=True)
class DeepEvidentialSegModelConfig(BaseConfig):
    # 特征提取函数的 config
    feature_extractor_config: FPNFeatureExtractorConfig = field(
        default_factory=FPNFeatureExtractorConfig
    )
    # classification head
    classification_head_channels: tuple[int, ...] = field(default=(128, 64, 32))
    num_classes: int = 25 # 在 data/RUGD/RUGD_annotations/RUGD_annotation-colormap.txt 中有说明
    # REMIND: 其他模型参数可以在这里添加


    def make_model(self, *, device='cpu') -> 'DeepEvidentialSegModel':
        model = DeepEvidentialSegModel(config=self)
        return model.to(device)

class DeepEvidentialSegModel(nn.Module):
    """Overall Model
    """
    def __init__(self, config: DeepEvidentialSegModelConfig):
        super().__init__()
        self.config = config
        # 特征提取器
        self.feature_extractor = config.feature_extractor_config.make_model()
        # 分类头
        self.classification_head = self._build_classification_head(
            in_channels=config.feature_extractor_config.decoder_segmentation_channels,
            hidden_channels=config.classification_head_channels,
            num_classes=config.num_classes
        )
        # REMIND: 这里可以添加更多的模块，例如 Normaliz flow, feature reconstruction 等
    
    def _build_classification_head(self, in_channels, hidden_channels, num_classes):
        """ 构建分类头
        """
        layers = []
        current_channels = in_channels
        for hidden_channel in hidden_channels:
            layers.append(nn.Linear(current_channels, hidden_channel))
            layers.append(nn.SELU(inplace=True))
            current_channels = hidden_channel
        layers.append(nn.Linear(current_channels, num_classes))
        layers.append(nn.Softmax(dim=-1))  # 假设我们需要概率输出
        return nn.Sequential(*layers)
    
    def get_current_device(self):
        return next(self.parameters()).device

    def classify(self, images, labels=None):
        """ 对输入图像进行分类，同时计算 loss
        """
        device= self.get_current_device()
        images = images.to(device)
        if labels is not None:
            labels = labels.to(device)

        loss = None

        *_, C, H, W = images.shape
        features = self.feature_extractor(images)
        features = features.permute(*range(len(_)), -2, -1, -3) # *CHW -> *HWC
        class_probs = self.classification_head(features)
        logits = torch.log(class_probs + 1e-8)  # 避免 log(0)

        if labels is not None:
            # 计算交叉熵损失
            one_hot_labels = nn.functional.one_hot(labels, num_classes=self.config.num_classes)
            loss = one_hot_labels * logits
            loss = -loss.sum(dim=-1).mean()  # 平均损失

        return loss, logits

if __name__ == "__main__":
    model = FPNFeatureExtractor(
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,                 # 此参数在提取特征时不再重要，但在初始化父类时是必须的
        decoder_segmentation_channels=128, # 这是你得到的特征向量的维度
    )