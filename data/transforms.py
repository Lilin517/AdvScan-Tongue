import torchvision.transforms as T

class MedicalAug:

    def __init__(self, input_size: int=224, train: bool=True):
        if train:
            self.transform = T.Compose([T.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)), T.RandomHorizontalFlip(p=0.3), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        else:
            self.transform = T.Compose([T.Resize(int(input_size * 1.14)), T.CenterCrop(input_size), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __call__(self, img):
        return self.transform(img)
