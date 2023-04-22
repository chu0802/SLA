import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler
from torchvision import transforms

from randaugment import RandAugmentMC

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = {
    "train": transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
    "strong": transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
}


def pil_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class ImageList(Dataset):
    def __init__(self, root, list_file, transform, strong_transform=None):
        with (root / list_file).open("r") as f:
            paths = [p[:-1].split() for p in f.readlines()]
        self.imgs = [(root / p, int(l)) for p, l in paths]
        self.transform = transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        img = pil_loader(path)
        if self.strong_transform:
            img1 = self.strong_transform(img)
            img2 = self.strong_transform(img)
            return self.transform(img), img1, img2, label
        return self.transform(img), label


class _InfiniteSampler(Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        worker_init_fn=None,
        generator=None,
        drop_last=True,
        num_workers=4,
    ):

        sampler = RandomSampler(dataset, replacement=False, generator=generator)
        batch_sampler = BatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last
        )

        self._infinite_iterator = iter(
            DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
                worker_init_fn=worker_init_fn,
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0


class DALoader:
    def __init__(self, args, domain_type, mode, labeled=None, strong_transform=False):
        self.seed = args.seed
        self.bsize = args.bsize
        self.domain_name = args.dataset["domains"][
            args.source if domain_type == "source" else args.target
        ]
        self.root = Path(args.dataset["path"])
        self.text_root = Path(args.dataset["text_path"])

        self.domain_type = domain_type
        self.mode = mode
        self.labeled = labeled
        self.num_shot = 3 if args.shot == "3shot" else 1
        self.strong_transform = strong_transform

    def get_loader_name(self):
        if self.labeled is None:
            return f"{self.domain_type}_{self.mode}"
        return f"{self.domain_type}_{self.labeled}_{self.mode}"

    def get_list_file_name(self):
        if self.domain_type == "source":
            return "all.txt"
        if self.mode == "validation":
            return "val.txt"
        if self.labeled == "labeled":
            return f"train_{self.num_shot}.txt"
        return f"test_{self.num_shot}.txt"

    def get_loader(self):
        list_file = self.text_root / self.domain_name / self.get_list_file_name()
        transform_mode = "train" if self.mode == "train" else "test"
        strong_transform = TRANSFORM["strong"] if self.strong_transform else None
        dset = ImageList(
            self.root,
            list_file,
            transform=TRANSFORM[transform_mode],
            strong_transform=strong_transform,
        )

        g = get_generator(self.seed)

        if self.mode == "train":
            dloader = InfiniteDataLoader(
                dset,
                batch_size=self.bsize,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=True,
                num_workers=4,
            )
        else:
            dloader = DataLoader(
                dset,
                batch_size=self.bsize,
                worker_init_fn=seed_worker,
                generator=g,
                shuffle=False,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
            )
        return dloader


class DataIterativeLoader:
    def __init__(self, args, strong_transform=False):
        required_loader = [
            DALoader(args, *types)
            for types in [
                ["source", "train"],
                ["source", "test"],
                ["target", "train", "labeled"],
                ["target", "test", "labeled"],
                ["target", "train", "unlabeled", strong_transform],
                ["target", "test", "unlabeled"],
                ["target", "validation"],
            ]
        ]

        self.loaders = {
            loader_type.get_loader_name(): loader_type.get_loader()
            for loader_type in required_loader
        }

        self.s_iter = iter(self.loaders["source_train"])
        self.l_iter = iter(self.loaders["target_labeled_train"])
        self.u_iter = iter(self.loaders["target_unlabeled_train"])

        self.strong_transform = strong_transform

    def __iter__(self):
        while True:
            sx, sy = next(self.s_iter)
            sx, sy = sx.float().cuda(), sy.long().cuda()

            tx, ty = next(self.l_iter)
            tx, ty = tx.float().cuda(), ty.long().cuda()

            if self.strong_transform:
                ux, ux1, ux2, _ = next(self.u_iter)
                ux = [ux.float().cuda(), ux1.float().cuda(), ux2.float().cuda()]
            else:
                ux, _ = next(self.u_iter)
                ux = [ux.float().cuda()]

            yield (sx, sy), (tx, ty), ux

    def __len__(self):
        return 0
