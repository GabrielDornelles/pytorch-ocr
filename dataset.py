import albumentations
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None, grayscale=False):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.grayscale = grayscale

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
            ]
        )
        if grayscale:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor()
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        if self.grayscale:
            image = self.transform(image)
        else:
            image = np.array(image)
            augmented = self.aug(image=image)
            image = augmented["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)

        return {
            "images": image, #
            "targets": torch.tensor(targets, dtype=torch.long),
        }
