import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2**32 - 1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def one_batch(dl):
    return next(iter(dl))


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    train_transforms = torchvision.transforms.Compose(
        [
            T.Resize(args.img_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )

    val_transforms = torchvision.transforms.Compose(
        [
            T.Resize(args.img_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )

    class BootstrappedImageFolder(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.image_folder = ImageFolder(root, transform=transform)
            self.class_to_idx = self.image_folder.class_to_idx
            self.samples = self.image_folder.samples
            self.targets = [s[1] for s in self.samples]

            # Organize samples by class
            self.class_samples = defaultdict(list)
            for idx, target in enumerate(self.targets):
                self.class_samples[target].append(idx)

            # Determine the max class size for bootstrapping
            self.max_class_size = max(len(samples) for samples in self.class_samples.values())

            # Create bootstrapped indices for each class
            self.bootstrap_indices = self._bootstrap_samples()

        def _bootstrap_samples(self):
            indices = []
            for class_id, samples in self.class_samples.items():
                # Resample with replacement to match max class size
                resampled = np.random.choice(samples, self.max_class_size, replace=True)
                indices.extend(resampled)
            return indices

        def __len__(self):
            return len(self.bootstrap_indices)

        def __getitem__(self, idx):
            actual_idx = self.bootstrap_indices[idx]
            return self.image_folder[actual_idx]

    # Usage example
    root = os.path.join(args.dataset_path, args.train_folder)
    train_dataset = BootstrappedImageFolder(root, transform=train_transforms)

    # train_dataset = ImageFolder(os.path.join(args.dataset_path, args.train_folder), transform=train_transforms)
    val_dataset = ImageFolder(os.path.join(args.dataset_path, args.val_folder), transform=val_transforms)

    if args.slice_size > 1:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
        val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), args.slice_size))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataset = DataLoader(
        val_dataset,
        batch_size=2 * args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_dataloader, val_dataset


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
