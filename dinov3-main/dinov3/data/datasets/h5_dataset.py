"""
H5Dataset is a custom dataset class for loading images from HDF5 (.h5/.hdf5) files.
适用于存储为 HDF5 格式的遥感图像数据。
"""

import os
import h5py
import numpy as np
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    """
    A custom dataset class for loading images from HDF5 files.

    Args:
        root (str): Root directory containing HDF5 files.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in a target and returns a transformed version.
        transforms (callable, optional): A function/transform that takes in both image and target and returns transformed versions.
        extensions (tuple): Supported file extensions.
        h5_key (str, optional): The key inside the h5 file where the image data is stored. 
                                If None, it will try to find the first 3D/2D dataset automatically.
    """

    def __init__(
            self, 
            root: str, 
            transform: Optional[Callable] = None, 
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            extensions: tuple = ('.h5', '.hdf5'),
            h5_key: Optional[str] = None,
        ):
        '''
        Args:
            root: 存放 HDF5 文件的文件夹路径
            transform: 图像预处理函数
            target_transform: 标签变换函数
            transforms: 同时对图像和标签进行变换的函数
            extensions: 支持的文件格式
            h5_key: HDF5 文件中存储图像数据的键名 (key)。如果为 None, 则自动尝试查找。
        '''
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.extensions = extensions
        self.h5_key = h5_key

        self.image_paths = self._load_image_paths(self.root)

    def _load_image_paths(self, root: str) -> List[str]:
        """Load all HDF5 file paths from the root directory."""
        image_paths = []
        if not os.path.exists(root): 
            raise FileNotFoundError(f"Root directory '{root}' does not exist.")
        
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(self.extensions):
                    image_paths.append(os.path.join(dirpath, filename))
        
        # # 保证每次加载顺序一致
        # image_paths.sort()
        return image_paths
    
    def __len__(self) -> int:
        """Return the number of HDF5 files in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Load and return an image and its corresponding target."""
        file_path = self.image_paths[idx]
        image = self._load_image(file_path)
        target = None  # 如果有标签，可以在这里根据需求加载

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target
    
    def _load_image(self, file_path: str) -> Image.Image:
        """
        Load an image from the given HDF5 path.
        Converts data to RGB PIL Image.
        """
        try:
            with h5py.File(file_path, 'r') as f:
                image_data = self._extract_image_data(f)
                
                if image_data is None:
                    raise ValueError(f"Could not find valid image data in {file_path}")

                # 处理数据类型和维度，确保转为 uint8 用于 PIL
                image_data = self._preprocess_array(image_data)
                
                return Image.fromarray(image_data, mode='RGB')
                
        except Exception as e:
            raise IOError(f"Error loading image from {file_path}: {str(e)}")

    def _extract_image_data(self, h5_file: h5py.File) -> Optional[np.ndarray]:
        """
        Extract image array from HDF5 file based on self.h5_key or auto-detection.
        """
        if self.h5_key:
            if self.h5_key in h5_file:
                return np.array(h5_file[self.h5_key])
            else:
                raise KeyError(f"Key '{self.h5_key}' not found in {h5_file.filename}")
        
        # 自动检测：寻找第一个合适维度的数据集
        def find_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                # 假设图像是 2D (H, W) 或 3D (H, W, C)
                if obj.ndim == 2 or (obj.ndim == 3 and obj.shape[2] in [1, 3, 4]):
                    return name
            return None

        # 遍历查找
        target_key = None
        h5_file.visititems(lambda name, obj: nonlocal_assign(name, obj) if target_key is None else None)
        
        # 由于 visititems 不支持简单的 break，这里用一种更直接的方式查找
        for key in h5_file.keys():
            item = h5_file[key]
            if isinstance(item, h5py.Dataset):
                if item.ndim == 2 or (item.ndim == 3 and item.shape[2] in [1, 3, 4]):
                    return np.array(item)
            elif isinstance(item, h5py.Group):
                # 如果第一层是组，可能需要递归，这里简化处理，只查第一层
                pass
        
        # 如果上面没找到，尝试递归查找第一个数据集
        found_key = None
        def visitor(name, obj):
            nonlocal found_key
            if isinstance(obj, h5py.Dataset) and found_key is None:
                 if obj.ndim == 2 or (obj.ndim == 3 and obj.shape[2] in [1, 3, 4]):
                     found_key = name
        
        h5_file.visititems(visitor)
        
        if found_key:
            return np.array(h5_file[found_key])
            
        return None

    def _preprocess_array(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess numpy array to be compatible with PIL Image.fromarray (H, W, 3) uint8.
        """
        # 1. 处理维度
        if image.ndim == 2:
            # 灰度图转 RGB
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                # (H, W, 1) -> (H, W, 3)
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] > 3:
                # 保留前三个通道
                image = image[:, :, :3]
            elif image.shape[0] == 3 or image.shape[1] == 3:
                # 可能是 (C, H, W) 格式，转为 (H, W, C)
                if image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                elif image.shape[1] == 3: # 这种情况较少见，视具体数据而定
                    image = np.transpose(image, (0, 2, 1))
        
        # 2. 处理数据类型
        # PIL 需要 uint8。如果数据是 float (0-1 或 0-255) 或其他类型，需要转换
        if image.dtype != np.uint8:
            if image.dtype == np.float32 or image.dtype == np.float64:
                # 假设浮点数范围是 0-1
                if image.max() <= 1.0:
                    image = image * 255.0
                # 如果已经是 0-255 范围的浮点数，直接转换
                image = image.clip(0, 255).astype(np.uint8)
            else:
                # 其他整数类型直接转 uint8 (可能涉及截断，需注意数据范围)
                image = image.astype(np.uint8)
                
        return image

# 辅助函数用于 nonlocal 赋值 (Python 闭包技巧，替代上面的 nonlocal_assign)
def nonlocal_assign(name, obj):
    pass