from torch.utils.data import Dataset
from glob import glob
from torchvision import transforms
from PIL import Image
import os


class SegmentationDataset(Dataset):
    """Custom dataset for binary segmentation."""

    def __init__(self, images_dir, masks_dir, img_size=(700, 500)):
        self.images = sorted(glob(os.path.join(images_dir, '*')))
        self.masks = sorted(glob(os.path.join(masks_dir, '*')))
        self.img_size = img_size

        # Transformations
        self.transform_img = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.Resampling.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Return a transformed image-mask pair."""
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        # Binarize mask (0 — background, 1 — object)
        mask = (mask > 0).float()
        return img, mask