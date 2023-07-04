import albumentations
from sklearn import preprocessing

import glob
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from sklearn import model_selection

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
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            ]
        )
        if grayscale:
            self.transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        if self.grayscale:
            image = self.transform(image)
        else:
            image = np.array(image)
            augmented = self.aug(image=image)
            image = augmented["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)

        return {
            "images": image,
            "targets": torch.tensor(targets, dtype=torch.long),
        }


def build_dataloaders(cfg):
    image_files = glob.glob(os.path.join(cfg.paths.dataset_dir, "*.png"))
    original_targets = [x.split("/")[-1][:-4].replace("-copy", "") for x in image_files]
    targets = [[c for c in x] for x in original_targets]
    targets_flat = [c for clist in targets for c in clist]
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(targets_flat)
    targets_encoded = [label_encoder.transform(x) for x in targets]
    targets_encoded = np.array(targets_encoded)
    targets_encoded = targets_encoded + 1

    (train_imgs, test_imgs, train_targets, test_targets, _, test_original_targets) = model_selection.train_test_split(
        image_files, targets_encoded, original_targets, test_size=0.1, random_state=42
    )

    train_dataset = ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(cfg.processing.image_height, cfg.processing.image_width),
        grayscale=cfg.model.gray_scale,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
    )

    test_dataset = ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(cfg.processing.image_height, cfg.processing.image_width),
        grayscale=cfg.model.gray_scale,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
    )
    return train_loader, test_loader, test_original_targets, label_encoder.classes_
