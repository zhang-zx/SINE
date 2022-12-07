import os
import numpy as np
import re
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from copy import deepcopy

import random
import torch

training_templates_smallest = [
    'painting of a sks {}',
]
style_templates_smallest = [
    '{} in the sks style',
]

reg_templates_smallest = [
    'painting of a {}',
]


reg_templates_smallest = [
    'photo of a {}',
]

reg_templates_no_class_smallest = [
    'a photo',
]

reg_templates_no_class_small = [
    'a photo',
    'a rendering',
    'a cropped photo',
    'the photo',
    'a dark photo',
    'a close-up photo',
    'a bright photo',
    'a cropped photo',
    'a good photo',
    'a rendition',
    'an illustration',
    'a depiction',
]


per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]


class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="dog",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 reg=False,
                 learn_style=False,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(
            self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text
        self.learn_style = learn_style

        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.reg = reg

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if not self.reg:
            if not self.learn_style:
                text = random.choice(training_templates_smallest).format(
                    placeholder_string)
            else:
                text = random.choice(style_templates_smallest).format(
                    placeholder_string)
        else:
            text = random.choice(reg_templates_smallest).format(
                placeholder_string)

        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class SinImageDataset(PersonalizedBase):
    def __init__(
        self,
        data_root,
        size=None,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="dog",
        per_image_tokens=False,
        center_crop=False,
        mixing_prob=0.25,
        coarse_class_text=None,
        reg=False,
        learn_style=False,
    ):
        self.data_root = data_root
        assert os.path.isfile(
            self.data_root), f"SinImageDataset requires a path to a image file, not a directory. Got {self.data_root}."

        self.image_paths = [self.data_root]*100

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text
        self.learn_style = learn_style

        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.reg = reg



class SinImageHighResDataset(Dataset):
    def __init__(self,
                 data_root,
                 size=512,
                 high_resolution=1024,
                 latent_scale=8,
                 min_crop_frac=0.5,
                 max_crop_frac=1.0,
                 rec_prob=0.0,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.,
                 set="train",
                 placeholder_token="dog",
                 per_image_tokens=False,

                 mixing_prob=0.25,
                 coarse_class_text=None):
        self.data_root = data_root
        assert os.path.isfile(
            self.data_root), f"SinImageDataset requires a path to a image file, not a directory. Got {self.data_root}."

        self.num_images = 100
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.high_resolution = high_resolution
        self.min_crop_frac = min_crop_frac
        self.max_crop_frac = max_crop_frac
        self.rec_prob = rec_prob
        self.latent_scale = latent_scale

        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        image = Image.open(self.data_root)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        self.image = image.resize(
            (self.high_resolution, self.high_resolution), resample=self.interpolation)

    def __len__(self):
        return self._length

    def _random_crop(self, pil_image):
        patch_size_y = int(
            (self.high_resolution//self.latent_scale) * (random.random() * (self.max_crop_frac - self.min_crop_frac) + self.min_crop_frac))
        patch_size_x = int(
            (self.high_resolution//self.latent_scale) * (random.random() * (self.max_crop_frac - self.min_crop_frac) + self.min_crop_frac))
        crop_y = random.randrange(
            (self.high_resolution//self.latent_scale) - patch_size_y + 1)
        crop_x = random.randrange(
            (self.high_resolution//self.latent_scale) - patch_size_x + 1)
        return pil_image.crop((crop_x * self.latent_scale, crop_y * self.latent_scale, (crop_x + patch_size_x) * self.latent_scale, (crop_y + patch_size_y) * self.latent_scale)).resize(
            (self.size, self.size),
            resample=self.interpolation), crop_y, crop_x, crop_y + patch_size_y, crop_x + patch_size_x

    def __getitem__(self, i):
        example = {}
        image = deepcopy(self.image)

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        text = random.choice(training_templates_smallest).format(
            placeholder_string)

        example["caption"] = text

        if random.random() < self.rec_prob:
            image, crop_y, crop_x, crop_y1, crop_x1 = self._random_crop(image)
            crop_area = torch.tensor([crop_y, crop_x, crop_y1, crop_x1])
        else:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)
            crop_area = torch.tensor(
                [0, 0, self.high_resolution//self.latent_scale, self.high_resolution//self.latent_scale])

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example['crop_boxes'] = crop_area
        return example
