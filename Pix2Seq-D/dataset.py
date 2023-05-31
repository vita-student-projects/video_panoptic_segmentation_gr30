from datasets.cityscapes import Cityscapes
from datasets.kittistep import KittiSTEP

import torch
from torchvision import datasets, transforms
from config import CFG
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


def get_pretrain_loaders_cityscapes():
    transform = A.Compose([
        A.ToFloat(max_value=255),
        A.augmentations.geometric.LongestMaxSize(
            max_size=CFG["image_size"].width, always_apply=True),
        A.RandomCrop(width=CFG["image_size"].width,
                     height=CFG["image_size"].height),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ],
        additional_targets={
            'target': 'image'
    }
    )

    trainloader = torch.utils.data.DataLoader(
        Cityscapes(
            "./data/Cityscapes",
            mode="fine",
            split="train",
            target_type="panoptic",
            frames_before=CFG["misc"]["frames_before"],
            transforms=transform
        ),
        batch_size=CFG["train"]["batch_size"],
        shuffle=True,
        num_workers=CFG["misc"]["num_workers"],
        pin_memory=True,
    )

    validloader = torch.utils.data.DataLoader(
        Cityscapes(
            "./data/Cityscapes",
            mode="fine",
            split="val",
            target_type="panoptic",
            frames_before=CFG["misc"]["frames_before"],
            transforms=transform
        ),
        batch_size=CFG["train"]["batch_size"],
        shuffle=False,
        # num_workers=CFG["misc"]["num_workers"],
        pin_memory=True,
    )

    return trainloader, validloader


def get_train_loaders_kittistep():
    transform = A.Compose([
        A.ToFloat(max_value=255),
        A.CenterCrop(width=CFG["image_size"].width,
                     height=CFG["image_size"].height),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ],
        additional_targets={
            'target': 'image'
    }
    )

    trainloader = torch.utils.data.DataLoader(
        KittiSTEP(
            # "./data/KittiSTEP",
            "./data/KittiSTEP",
            mode="train",
            download=True,
            frames_before=CFG["misc"]["frames_before"],
            transforms=transform
        ),
        batch_size=CFG["train"]["batch_size"],
        shuffle=True,
        # num_workers=CFG["misc"]["num_workers"],
        pin_memory=True,
    )

    validloader = torch.utils.data.DataLoader(
        KittiSTEP(
            "./data/KittiSTEP",
            mode="val",
            frames_before=CFG["misc"]["frames_before"],
            transforms=transform
        ),
        batch_size=CFG["train"]["batch_size"],
        shuffle=False,
        # num_workers=CFG["misc"]["num_workers"],
        pin_memory=True,
    )

    return trainloader, validloader
