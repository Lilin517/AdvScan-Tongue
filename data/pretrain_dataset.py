import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class UnlabeledImageDataset(Dataset):

    def __init__(self, root_dir: str, transform=None, extensions=('.jpg', '.jpeg', '.png')):
        self.paths = []
        for ext in extensions:
            self.paths.extend(glob.glob(os.path.join(root_dir, '**', f'*{ext}'), recursive=True))
        self.paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
