import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class DeepLabSegmentationDataset(Dataset):
    """
    用于训练 DeepLabV3+ 的遥感语义分割数据集
    支持：图像归一化、掩码清理、transform增强、错误检查
    """
    def __init__(self, image_dir, mask_dir, list_path, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 读取图像列表
        with open(list_path, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, img_id + ".png")
        mask_path = os.path.join(self.mask_dir, img_id + ".png")
        img_name = img_id  # 直接使用图像 ID 作为图名（不带扩展名）

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码文件: {mask_path}")

        # 预处理或增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)
            mask = torch.from_numpy(mask).long()

        return image, mask, img_name

