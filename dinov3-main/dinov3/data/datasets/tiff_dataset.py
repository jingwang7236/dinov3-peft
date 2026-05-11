"""
TiffDataset is a custom dataset class for loading TIFF images. 
适用于遥感图像等需要支持.tif/.tiff格式的图片, 仅支持RGB3通道图像
"""

import os
from typing import Any, Callable, List, Optional, Tuple, Union
from PIL import Image
from torch.utils.data import Dataset

import tifffile
import numpy as np

class TiffDataset(Dataset):
    """
    A custom dataset class for loading TIFF images.

    Args:
        root_dir (str): Root directory containing TIFF images.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in a target and returns a transformed version.
    """

    def __init__(
            self, 
            root: str, 
            transform: Optional[Callable] = None, 
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            extensions: tuple = ('.tif', '.tiff', '.png', '.jpg', '.jpeg'),
        ):
        '''
        Args:
            root: 存放图像的文件夹路径
            transform: 图像预处理函数
            target_transform: 标签变换函数
            transforms: 同时对图像和标签进行变换的函数, 通常用于DataAugmentationDINO 
            extensions: 支持的文件格式
        '''
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.extensions = extensions

        self.image_paths = self._load_image_paths(self.root)

    def _load_image_paths(self, root: str) -> List[str]:
        """Load all image file paths from the root directory."""
        image_paths = []
        if not os.path.exists(root): 
            raise FileNotFoundError(f"Root directory '{root}' does not exist.")
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(self.extensions):
                    image_paths.append(os.path.join(dirpath, filename))
        return image_paths
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Load and return an image and its corresponding target."""
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        target = None  # 如果有标签, 可以在这里加载

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from the given path.
            PNG or TIFF format is supported.
        """
        if image_path.lower().endswith(('.tif', '.tiff')):
            image = tifffile.imread(image_path)
            if image.ndim == 2:  # 灰度图像, 转为RGB
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] > 3:  # 多通道图像, 仅保留前三个通道
                image = image[:, :, :3]
            image = Image.fromarray(image, mode='RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image