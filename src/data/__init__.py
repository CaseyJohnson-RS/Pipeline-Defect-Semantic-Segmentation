from .augmentation import augmentation_segmentation_ds  # noqa: F401
from .SegmentationDataset import SegmentationDataset  # noqa: F401
import os


def check_dataset_dirs(dataset_path: str) -> bool:
    for data_dir in ["images", "masks"]:
        for divide_dir in ["train", "val"]:
            if not os.path.isdir(
                os.path.join("datasets", dataset_path, data_dir, divide_dir)
            ):
                return False
    return True
