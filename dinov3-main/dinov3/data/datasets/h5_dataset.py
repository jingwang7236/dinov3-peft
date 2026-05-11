"""
H5Dataset implementation following the style of ImageNet in dinov3.
Supports Train/Val/Test splits via subdirectories.
"""

import os
import logging
import h5py
import numpy as np
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset

# Import base class and decoders if available in the project structure
# Assuming standard DINOv3 structure:
try:
    from .extended import ExtendedVisionDataset
    from .decoders import ImageDataDecoder, TargetDecoder
except ImportError:
    # Fallback for standalone testing or if structure differs
    class ExtendedVisionDataset(Dataset):
        def __init__(self, root, transforms=None, transform=None, target_transform=None, image_decoder=None, target_decoder=None):
            self.root = root
            self.transforms = transforms
            self.transform = transform
            self.target_transform = target_transform
            self.image_decoder = image_decoder
            self.target_decoder = target_decoder
            
    class ImageDataDecoder:
        def __call__(self, data):
            return data # Placeholder

logger = logging.getLogger("dinov3")


class _Split(Enum):
    """
    Enum for dataset splits: TRAIN, VAL, TEST.
    """
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def dirname(self) -> str:
        return self.value


class H5DataDecoder:
    """
    Decodes image data from HDF5 files.
    Compatible with the decoder interface used in ExtendedVisionDataset.
    """
    def __init__(self, h5_key: Optional[str] = None):
        self.h5_key = h5_key

    def __call__(self, file_path: str) -> Image.Image:
        """
        Load and decode an image from an HDF5 file.
        
        Args:
            file_path: Path to the .h5 file.
            
        Returns:
            PIL Image in RGB mode.
        """
        try:
            with h5py.File(file_path, 'r') as f:
                image_data = self._extract_image_data(f)
                if image_data is None:
                    raise ValueError(f"Could not find valid image data in {file_path}")
                
                # Preprocess array to uint8 RGB
                image_array = self._preprocess_array(image_data)
                return Image.fromarray(image_array, mode='RGB')
        except Exception as e:
            raise IOError(f"Error loading image from {file_path}: {str(e)}")

    def _extract_image_data(self, h5_file: h5py.File) -> Optional[np.ndarray]:
        """Extract image array based on key or auto-detection."""
        if self.h5_key:
            if self.h5_key in h5_file:
                return np.array(h5_file[self.h5_key])
            else:
                logger.warning(f"Key '{self.h5_key}' not found in {h5_file.filename}, attempting auto-detection.")
        
        # Auto-detection logic
        found_key = None
        
        def visitor(name, obj):
            nonlocal found_key
            if found_key is not None:
                return # Stop if already found
            if isinstance(obj, h5py.Dataset):
                # Check for typical image shapes: (H, W), (H, W, 1), (H, W, 3), (H, W, 4)
                # Or channel-first: (1, H, W), (3, H, W)
                if obj.ndim == 2:
                    found_key = name
                elif obj.ndim == 3:
                    if obj.shape[2] in [1, 3, 4] or obj.shape[0] in [1, 3, 4]:
                        found_key = name

        h5_file.visititems(visitor)
        
        if found_key:
            return np.array(h5_file[found_key])
        return None

    def _preprocess_array(self, image: np.ndarray) -> np.ndarray:
        """Convert numpy array to uint8 RGB (H, W, 3)."""
        # Handle Channel First vs Channel Last
        if image.ndim == 3:
            if image.shape[0] in [1, 3, 4] and image.shape[2] not in [1, 3, 4]:
                # Likely (C, H, W) -> (H, W, C)
                image = np.transpose(image, (1, 2, 0))
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
        
        # Handle dtype
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                if image.max() <= 1.0:
                    image = image * 255.0
                image = image.clip(0, 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        return image


class H5Dataset(ExtendedVisionDataset):
    """
    Dataset for loading images from HDF5 files, structured similarly to ImageNet.
    
    Directory Structure Expected:
        root/
        ├── train/
        │   ├── sample1.h5
        │   └── ...
        ├── val/
        │   ├── sample1.h5
        │   └── ...
        └── test/
            ├── sample1.h5
            └── ...
    """
    
    Split = _Split

    def __init__(
        self,
        *,
        split: "H5Dataset.Split",
        root: str,
        extra: Optional[str] = None, # Kept for interface consistency with ImageNet
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.h5', '.hdf5'),
        h5_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=H5DataDecoder(h5_key=h5_key),
            target_decoder=None, # No targets for self-supervised pre-training
        )
        
        self.split = split
        self.extensions = extensions
        self._extra_root = extra
        
        # Load entries (file paths) for the specific split
        self._entries = self._load_entries()

    def _load_entries(self) -> np.ndarray:
        """
        Scan the split directory and return a sorted array of file paths.
        Using numpy array for consistency with ImageNet's entry handling.
        """
        split_dir = os.path.join(self.root, self.split.dirname)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Split directory '{split_dir}' does not exist. "
                f"Please ensure your data is organized as {self.root}/train/, {self.root}/val/, etc."
            )
        
        file_paths = []
        for dirpath, _, filenames in os.walk(split_dir):
            for filename in filenames:
                if filename.lower().endswith(self.extensions):
                    full_path = os.path.join(dirpath, filename)
                    file_paths.append(full_path)
        
        # Sort to ensure deterministic order
        # file_paths.sort()
        
        if not file_paths:
            raise ValueError(f"No H5 files found in {split_dir} with extensions {self.extensions}")
            
        logger.info(f"Found {len(file_paths)} H5 files in split '{self.split.value}'.")
        
        # Convert to numpy array of strings for consistency
        return np.array(file_paths, dtype=str)

    def __len__(self) -> int:
        return len(self._entries)

    def get_image_data(self, index: int) -> str:
        """
        Return the file path for the given index.
        The H5DataDecoder will use this path to load and decode the image.
        Note: In standard ImageNet, this returns bytes. Here we return path 
        because H5 decoding requires random access within the file which is 
        harder to do with raw bytes passed from here without seeking logic.
        The __getitem__ in ExtendedVisionDataset might need to be aware of this,
        or we override __getitem__ to use the decoder directly with the path.
        
        For strict compatibility with ExtendedVisionDataset which expects 
        decoder(image_bytes), we might need to adjust. However, custom decoders 
        can handle paths if the base class allows. 
        
        If the base class strictly calls decoder(get_image_data(idx)), and 
        decoder expects bytes, this approach fails. 
        Let's override __getitem__ to ensure correct flow for H5.
        """
        return self._entries[index]

    def get_target(self, index: int) -> Optional[Any]:
        """No targets for self-supervised pre-training."""
        return None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Override to handle H5 path-based decoding correctly.
        """
        file_path = self._entries[index]
        
        # Use the decoder directly with the file path
        try:
            image = self.image_decoder(file_path)
        except Exception as e:
            logger.error(f"Failed to load index {index} at {file_path}: {e}")
            raise e

        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target