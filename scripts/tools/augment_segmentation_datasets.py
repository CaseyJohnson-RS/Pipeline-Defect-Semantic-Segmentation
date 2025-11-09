import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)


from src.datasets import augmentation_segmentation_ds  # noqa: E402
from src.tools import check_dataset_dirs # noqa: E402


if __name__ == "__main__":
    
    datasets = [
        "PipeSegmentation",
        "PipeBoxSegmentation"
    ]

    for ds in datasets:

        if check_dataset_dirs(ds):
            augmentation_segmentation_ds(ds, n_aug=3, seed=42)
        else:
            print(f"ERROR: check dataset directory structure {ds}!")

