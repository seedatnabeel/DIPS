from PIL import Image
from torchvision import datasets, transforms
from utils.randaugment import RandAugmentMC


class TransformFixMatch(object):
    def __init__(self, mean, std, crop_size=32, size_image=32, useCutout=True):
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=crop_size,
                    padding=int(size_image * 0.125),
                    padding_mode="reflect",
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=crop_size,
                    padding=int(size_image * 0.125),
                    padding_mode="reflect",
                ),
                RandAugmentMC(n=2, m=10, useCutout=useCutout),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        if not isinstance(x, Image.Image):
            x = transforms.ToPILImage()(x)
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
