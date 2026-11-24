import glob
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class FacadesDataset(Dataset):
    def __init__(self, root, split="train", direction="BtoA", load_size=286, crop_size=256):
        self.paths = sorted(glob.glob(os.path.join(root, split, "*.jpg")))
        if not self.paths:
            raise FileNotFoundError(f"No .jpg files found in {os.path.join(root, split)}")
        self.direction = direction
        self.train = split == "train"

        self.augment = transforms.Compose([
            transforms.Resize((load_size, load_size), antialias=True),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5] * 6, std=[0.5] * 6)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        w, h = img.size
        half = w // 2
        a = img.crop((0, 0, half, h))
        b = img.crop((half, 0, w, h))

        if self.direction == "BtoA":
            a, b = b, a

        pair = torch.cat([self.to_tensor(a), self.to_tensor(b)], dim=0) # (6, h, w)

        if self.train:
            pair = self.augment(pair)
        pair = self.normalize(pair)

        return pair[:3], pair[3:]
