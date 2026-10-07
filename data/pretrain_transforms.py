import random
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class GaussianBlur:

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = TF.gaussian_blur(x, kernel_size=23, sigma=sigma)
        return x

class Solarization:

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        if random.random() < self.threshold:
            return TF.solarize(x, threshold=128)
        return x

class DINOMultiCrop:

    def __init__(self, global_crops_size: int=224, local_crops_size: int=96, global_crops_scale: tuple=(0.4, 1.0), local_crops_scale: tuple=(0.05, 0.4), local_crops_number: int=6, resize_local_to_global: bool=False):
        color_jitter = T.ColorJitter(0.3, 0.3, 0.15, 0)
        self.global_transform = T.Compose([T.RandomResizedCrop(global_crops_size, scale=global_crops_scale), T.RandomHorizontalFlip(p=0.3), color_jitter, GaussianBlur(sigma=(0.1, 1.0)), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        local_output_size = global_crops_size if resize_local_to_global else local_crops_size
        self.local_transform = T.Compose([T.RandomResizedCrop(local_output_size, scale=local_crops_scale), T.RandomHorizontalFlip(p=0.3), color_jitter, GaussianBlur(sigma=(0.1, 0.5)), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.local_crops_number = local_crops_number

    def __call__(self, img):
        global_crops = [self.global_transform(img) for _ in range(2)]
        local_crops = [self.local_transform(img) for _ in range(self.local_crops_number)]
        return (global_crops, local_crops)

class BYOLTransform:

    def __init__(self, input_size: int=224):
        self.transform = T.Compose([T.RandomResizedCrop(input_size, scale=(0.4, 1.0)), T.RandomHorizontalFlip(p=0.3), T.ColorJitter(0.3, 0.3, 0.15, 0), GaussianBlur(sigma=(0.1, 1.0)), Solarization(threshold=0.1), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __call__(self, img):
        view1 = self.transform(img)
        view2 = self.transform(img)
        return (view1, view2)
