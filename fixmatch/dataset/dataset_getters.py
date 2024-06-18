# Code adapted from the repository https://github.com/UCSC-REAL/cifar-10-100n

import logging
import math

import medmnist
import numpy as np
import torch
from dcai.curation import train_and_clean
from medmnist import INFO
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from utils.constants import STATS
from utils.datasets import CustomDataset, TransformedSubset, x_u_split
from utils.transforms import TransformFixMatch

from .cifarn import CIFAR10SSL, CIFAR100SSL, CIFAR10n

logger = logging.getLogger(__name__)

cifar10_mean = STATS["cifar10 mean"]
cifar10_std = STATS["cifar10 std"]
cifar100_mean = STATS["cifar100 mean"]
cifar100_std = STATS["cifar100 std"]
normal_mean = STATS["normal mean"]
normal_std = STATS["normal std"]


def get_cifar10(args):
    noise_path = "../dataset/CIFAR-10_human.pt"
    # Define the transforms
    train_cifar10_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_cifar10_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ]
    )

    base_dataset = CIFAR10n(
        root="../dataset",
        download=True,
        train=True,
        transform=train_cifar10_transform,
        noise_type=args.noise_type,
        noise_path=noise_path,
        is_human=True,
    )

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    unlabeled_size = int(len(train_unlabeled_idxs) * args.unlabeled_prop)
    train_unlabeled_idxs = np.random.choice(
        train_unlabeled_idxs, size=unlabeled_size, replace=True, p=None
    )

    unique_indices = np.unique(train_labeled_idxs)

    train_labeled_dataset = CIFAR10SSL(
        root="../dataset",
        download=True,
        train=True,
        transform=train_cifar10_transform,
        noise_type=args.noise_type,
        noise_path=noise_path,
        is_human=True,
        indexs=train_labeled_idxs,
    )

    train_unlabeled_dataset = CIFAR10SSL(
        root="../dataset",
        download=True,
        train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        noise_type=args.noise_type,
        noise_path=noise_path,
        is_human=True,
        indexs=train_unlabeled_idxs,
    )

    dips_labeled_dataset = CIFAR10SSL(
        root="../dataset",
        download=True,
        train=True,
        transform=transform_val,
        noise_type=args.noise_type,
        noise_path=noise_path,
        is_human=True,
        indexs=unique_indices,
    )

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    # Dataloader creation
    labeled_trainloader = DataLoader(
        train_labeled_dataset,
        sampler=train_sampler(train_labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unshuffled_labeled_trainloader = DataLoader(
        dips_labeled_dataset,
        batch_size=500,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Train the model
    if args.use_dips or args.use_small_loss:
        indices_easy, _ = train_and_clean(
            args,
            labeled_trainloader,
            unshuffled_labeled_trainloader,
        )
    else:
        indices_easy = np.arange(len(train_labeled_idxs))

    keep_ids = unique_indices[indices_easy]

    if len(keep_ids) < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * 2 / len(keep_ids))
        keep_ids = np.hstack([keep_ids for _ in range(num_expand_x)])

    train_labeled_dataset = CIFAR10SSL(
        root="../dataset",
        download=True,
        train=True,
        transform=train_cifar10_transform,
        noise_type=args.noise_type,
        noise_path=noise_path,
        is_human=True,
        indexs=keep_ids,
    )

    test_dataset = CIFAR10n(
        root="../dataset",
        download=True,
        train=False,
        transform=test_cifar10_transform,
        is_human=True,
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_tissuemnist(args):
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    size_image = 28
    crop_size = 28

    train_tissue_transform = transforms.Compose(
        [
            transforms.RandomCrop(size_image, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    test_tissue_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    data_flag = "tissuemnist"
    download = True
    info = INFO[data_flag]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])
    args.num_classes = n_classes

    trainset = DataClass(
        split="train", transform=train_tissue_transform, download=download
    )
    testset = DataClass(
        split="test", transform=test_tissue_transform, download=download
    )

    trainset.labels = trainset.labels.flatten()
    testset.labels = testset.labels.flatten()

    # Accessing the targets
    train_targets = trainset.labels.flatten()

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, train_targets)

    unlabeled_size = int(len(train_unlabeled_idxs) * args.unlabeled_prop)
    train_unlabeled_idxs = np.random.choice(
        train_unlabeled_idxs, size=unlabeled_size, replace=True, p=None
    )

    unique_indices = np.unique(train_labeled_idxs)

    n_mislabeled = int(len(train_labeled_idxs) * args.prop_label_noise)

    # Randomly select indices to mislabel
    mislabeled_indices = np.random.choice(
        train_labeled_idxs, n_mislabeled, replace=False
    )

    # Create a copy of the targets to modify
    modified_targets = np.array(train_targets)

    for k, idx in enumerate(mislabeled_indices):
        current_label = train_targets[idx]
        new_label = np.random.choice([l for l in range(8) if l != current_label])
        modified_targets[idx] = new_label

    train_labeled_dataset = CustomDataset(
        trainset, train_labeled_idxs, modified_targets[train_labeled_idxs]
    )
    dips_labeled_dataset = CustomDataset(
        trainset, train_labeled_idxs, modified_targets[train_labeled_idxs]
    )

    tissue_mean = [0.5]
    tissue_std = [0.5]
    train_unlabeled_dataset = TransformedSubset(
        trainset,
        train_unlabeled_idxs,
        transform=TransformFixMatch(
            mean=tissue_mean,
            std=tissue_std,
            crop_size=crop_size,
            size_image=size_image,
            useCutout=False,
        ),
    )

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    # Dataloader creation
    labeled_trainloader = DataLoader(
        train_labeled_dataset,
        sampler=train_sampler(train_labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unshuffled_labeled_trainloader = DataLoader(
        dips_labeled_dataset,
        batch_size=500,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Train the model
    if args.use_dips or args.use_small_loss:
        indices_easy, _ = train_and_clean(
            args,
            labeled_trainloader,
            unshuffled_labeled_trainloader,
        )
    else:
        indices_easy = np.arange(len(train_labeled_idxs))

    keep_ids = unique_indices[indices_easy]

    if len(keep_ids) < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * 2 / len(keep_ids))
        keep_ids = np.hstack([keep_ids for _ in range(num_expand_x)])

    train_labeled_dataset = CustomDataset(
        trainset, keep_ids, modified_targets[keep_ids]
    )

    test_dataset = testset

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_pathmnist(args):

    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    size_image = 28
    crop_size = 28

    train_tissue_transform = transforms.Compose(
        [
            transforms.RandomCrop(size_image, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    test_tissue_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    data_flag = "pathmnist"
    download = True
    info = INFO[data_flag]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])
    args.num_classes = n_classes

    trainset = DataClass(
        split="train", transform=train_tissue_transform, download=download
    )
    testset = DataClass(
        split="test", transform=test_tissue_transform, download=download
    )

    trainset.labels = trainset.labels.flatten()
    testset.labels = testset.labels.flatten()

    # Accessing the targets
    train_targets = trainset.labels.flatten()

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, train_targets)

    unlabeled_size = int(len(train_unlabeled_idxs) * args.unlabeled_prop)
    train_unlabeled_idxs = np.random.choice(
        train_unlabeled_idxs, size=unlabeled_size, replace=True, p=None
    )

    unique_indices = np.unique(train_labeled_idxs)

    n_mislabeled = int(len(train_labeled_idxs) * args.prop_label_noise)

    # Randomly select indices to mislabel
    mislabeled_indices = np.random.choice(
        train_labeled_idxs, n_mislabeled, replace=False
    )

    # Create a copy of the targets to modify
    modified_targets = np.array(train_targets)

    for k, idx in enumerate(mislabeled_indices):
        current_label = train_targets[idx]
        new_label = np.random.choice([l for l in range(9) if l != current_label])
        modified_targets[idx] = new_label

    train_labeled_dataset = CustomDataset(
        trainset, train_labeled_idxs, modified_targets[train_labeled_idxs]
    )
    dips_labeled_dataset = CustomDataset(
        trainset, train_labeled_idxs, modified_targets[train_labeled_idxs]
    )

    tissue_mean = [0.5]
    tissue_std = [0.5]
    train_unlabeled_dataset = TransformedSubset(
        trainset,
        train_unlabeled_idxs,
        transform=TransformFixMatch(
            mean=tissue_mean,
            std=tissue_std,
            crop_size=crop_size,
            size_image=size_image,
            useCutout=False,
        ),
    )

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    # Dataloader creation
    labeled_trainloader = DataLoader(
        train_labeled_dataset,
        sampler=train_sampler(train_labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unshuffled_labeled_trainloader = DataLoader(
        dips_labeled_dataset,
        batch_size=500,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Train the model
    if args.use_dips or args.use_small_loss:
        indices_easy, _ = train_and_clean(
            args,
            labeled_trainloader,
            unshuffled_labeled_trainloader,
        )
    else:
        indices_easy = np.arange(len(train_labeled_idxs))

    keep_ids = unique_indices[indices_easy]

    if len(keep_ids) < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * 2 / len(keep_ids))
        keep_ids = np.hstack([keep_ids for _ in range(num_expand_x)])

    train_labeled_dataset = CustomDataset(
        trainset, keep_ids, modified_targets[keep_ids]
    )

    test_dataset = testset

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_eurosat(args):
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    train_eurosat_transform = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    from torchvision.datasets import EuroSAT

    base_dataset = EuroSAT(
        root="../dataset", transform=train_eurosat_transform, download=True
    )
    trainset, test_set = torch.utils.data.random_split(base_dataset, [21600, 5400])

    # Accessing the targets
    train_targets = np.array([base_dataset.targets[i] for i in trainset.indices])

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, train_targets)

    unlabeled_size = int(len(train_unlabeled_idxs) * args.unlabeled_prop)
    train_unlabeled_idxs = np.random.choice(
        train_unlabeled_idxs, size=unlabeled_size, replace=True, p=None
    )

    unique_indices = np.unique(train_labeled_idxs)

    n_mislabeled = int(len(train_labeled_idxs) * args.prop_label_noise)

    # Create a copy of the targets to modify
    modified_targets = np.array(train_targets)

    # Randomly select indices to mislabel
    mislabeled_indices = np.random.choice(
        train_labeled_idxs, n_mislabeled, replace=False
    )

    for k, idx in enumerate(mislabeled_indices):
        current_label = train_targets[idx]
        new_label = np.random.choice([l for l in range(10) if l != current_label])
        modified_targets[idx] = new_label

    train_labeled_dataset = CustomDataset(
        trainset, train_labeled_idxs, modified_targets[train_labeled_idxs]
    )
    dips_labeled_dataset = CustomDataset(
        trainset, train_labeled_idxs, modified_targets[train_labeled_idxs]
    )

    train_unlabeled_dataset = TransformedSubset(
        trainset,
        train_unlabeled_idxs,
        transform=TransformFixMatch(
            mean=normal_mean, std=normal_std, crop_size=64, size_image=64
        ),
    )

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    # Dataloader creation
    labeled_trainloader = DataLoader(
        train_labeled_dataset,
        sampler=train_sampler(train_labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unshuffled_labeled_trainloader = DataLoader(
        dips_labeled_dataset,
        batch_size=500,
        num_workers=args.num_workers,
        shuffle=False,
    )

    if args.use_dips or args.use_small_loss:
        indices_easy, _ = train_and_clean(
            args,
            labeled_trainloader,
            unshuffled_labeled_trainloader,
        )
        keep_ids = unique_indices[indices_easy]

    else:
        indices_easy = np.arange(len(train_labeled_idxs))
        keep_ids = unique_indices[indices_easy]

    if len(keep_ids) < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * 2 / len(keep_ids))
        keep_ids = np.hstack([keep_ids for _ in range(num_expand_x)])

    train_labeled_dataset = CustomDataset(
        trainset, keep_ids, modified_targets[keep_ids]
    )

    test_dataset = test_set

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args):
    root = "../dataset"

    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ]
    )

    base_dataset = datasets.CIFAR100(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR100SSL(
        root,
        train_unlabeled_idxs,
        train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std),
    )

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


DATASET_GETTERS = {
    "cifar10": get_cifar10,
    "cifar100": get_cifar100,
    "eurosat": get_eurosat,
    "tissue": get_tissuemnist,
    "pathmnist": get_pathmnist,
}
