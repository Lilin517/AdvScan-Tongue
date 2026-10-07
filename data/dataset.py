import os
import json
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset

class COCOMultiLabelDataset(Dataset):

    def __init__(self, root_dir: str, split: str='train', transform=None, classes_txt: str=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        ann_file = os.path.join(root_dir, split, 'annotations', f'{split}.json')
        img_dir = os.path.join(root_dir, split, 'images')
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f'Annotation file not found: {ann_file}')
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        self.categories = []
        if classes_txt and os.path.isfile(classes_txt):
            with open(classes_txt, 'r', encoding='utf-8') as f:
                self.categories = [line.strip() for line in f if line.strip()]
        else:
            cats_sorted = sorted(coco_data.get('categories', []), key=lambda c: c['id'])
            self.categories = [c['name'] for c in cats_sorted]
        self.num_classes = len(self.categories)
        self.cat_id_to_idx = {c['id']: i for i, c in enumerate(sorted(coco_data.get('categories', []), key=lambda c: c['id']))}
        img_to_labels = defaultdict(set)
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            cat_idx = self.cat_id_to_idx.get(ann['category_id'])
            if cat_idx is not None:
                img_to_labels[img_id].add(cat_idx)
        self.samples = []
        for img_info in coco_data.get('images', []):
            img_id = img_info['id']
            file_name = img_info['file_name']
            img_path = os.path.join(img_dir, file_name)
            if not os.path.isfile(img_path):
                continue
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            for cat_idx in img_to_labels.get(img_id, set()):
                label[cat_idx] = 1.0
            self.samples.append((img_path, label))
        self.samples.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return (img, label)
