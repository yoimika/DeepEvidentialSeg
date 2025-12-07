from typing import List, Set
import torch
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# RUGD 数据集的标签颜色映射
# 颜色格式为 RGB
label2color = {
    0: (0, 0, 0),  # void
    1: (108, 64, 20),  # dirt
    2: (255, 229, 204),  # sand
    3: (0, 102, 0),  # grass
    4: (0, 255, 0),  # tree
    5: (0, 153, 153),  # pole
    6: (0, 128, 255),  # water
    7: (0, 0, 255),  # sky
    8: (255, 255, 0),  # vehicle
    9: (255, 0, 127),  # container/generic-object
    10: (64, 64, 64),  # asphalt
    11: (255, 128, 0),  # gravel
    12: (255, 0, 0),   # building
    13: (153, 76, 0),   # mulch
    14: (102, 102, 0),   # rock-bed
    15: (102, 0, 0),   # log
    16: (0, 255, 128),   # bicycle
    17: (204, 153, 255),   # person
    18: (102, 0, 204),   # fence
    19: (255,153 ,204),   # bush
    # 以下标签在 RUGD 数据集中不存在
    20: (0, 102, 102),  # sign
    21: (153, 204, 255),  # rock
    22: (102, 255, 255),  # bridge
    23: (101, 101, 11),  # concrete
    24: (114, 85, 47)   # picnic-table
}

label2name = {
    0: 'void',
    1: 'dirt',
    2: 'sand',
    3: 'grass',
    4: 'tree',
    5: 'pole',
    6: 'water',
    7: 'sky',
    8: 'vehicle',
    9: 'container/generic-object',
    10: 'asphalt',
    11: 'gravel',
    12: 'building',
    13: 'mulch',
    14: 'rock-bed',
    15: 'log',
    16: 'bicycle',
    17: 'person',
    18: 'fence',
    19: 'bush',
    20: 'sign',
    21: 'rock',
    22: 'bridge',
    23: 'concrete',
    24: 'picnic-table'
}


def _channel_adjust(image: torch.Tensor) -> torch.Tensor:
    # 将图像调整为 C x H x W 格式
    if isinstance(image, torch.Tensor) and image.dim() == 3:
        if image.shape[0] == 3:
            return image
        else:
            return image.permute(2, 0, 1)  # HWC to CHW
    else:
        # Expect B x C x H x W
        if image.shape[1] == 3:
            return image
        else:
            return image.permute(0, 3, 1, 2)

def _random_crop(image: torch.Tensor, label: torch.Tensor) -> tuple:
    *_, H, W = image.shape
    crop_h, crop_w = H // 32 * 32, W // 32 * 32  # 确保裁剪尺寸是 32 的倍数

    start_h = np.random.randint(0, H - crop_h + 1)
    start_w = np.random.randint(0, W - crop_w + 1)

    image = image[..., start_h:start_h + crop_h, start_w:start_w + crop_w]
    label = label[..., start_h:start_h + crop_h, start_w:start_w + crop_w]
    return image, label

def image_transforms(image: object) -> object:
    # 标准化图像
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    output = transforms(image)
    return _channel_adjust(output)

def image_reverse_transforms(image: object) -> object:
    # 反标准化图像
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    output = inv_normalize(image)
    output = torch.clamp(output, 0, 1)
    return _channel_adjust(output)

def map_label_colors(label: torch.Tensor) -> torch.Tensor:
    # 将 RGB 颜色映射到标签 ID
    for id, color in label2color.items():
        mask = (label == torch.tensor(color)).all(dim=-1)
        label[mask] = id
    return label

def label_transforms(label: object) -> object:    
    output = map_label_colors(torch.from_numpy(label)).long()
    output = _channel_adjust(output)
    if output.dim() == 3:
        return output[0]
    else:
        return output[:, 0]

def is_label_in_image(label: torch.Tensor, target_labels: Set[int]) -> bool:
    # 检查图像中是否包含目标标签
    unique_labels = torch.unique(label)
    for t_label in target_labels:
        if t_label in unique_labels:
            return True
    return False

def visualize_with_legend(ax, mask, title, label2name=label2name):
    # 1. Get unique classes present in this specific mask
    unique_labels = np.unique(mask)
    
    # 2. Generate a colormap with enough colors for the max label index (25)
    # using 'tab20' or 'jet' usually provides distinct colors
    base_cmap = plt.get_cmap('tab20b', 25) 
    
    # 3. Plot the image
    im = ax.imshow(mask, cmap=base_cmap, vmin=0, vmax=24)
    ax.set_title(title)
    
    # 4. Create the legend handles manually
    legend_patches = []
    for label_idx in unique_labels:
        if label_idx in label2name:
            # Get the color corresponding to this label from the colormap
            color = base_cmap(label_idx / 24.0) 
            patch = mpatches.Patch(color=color, label=f"{label_idx}: {label2name[label_idx]}")
            legend_patches.append(patch)
    
    # 5. Add legend to the side of the plot
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)