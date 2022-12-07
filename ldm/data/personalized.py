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
    'photo of a sks {}',
]

reg_templates_smallest = [
    'photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
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
                 reg = False
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

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
            text = random.choice(training_templates_smallest).format(placeholder_string)
        else:
            text = random.choice(reg_templates_smallest).format(placeholder_string)
            
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
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example

def crop_image(img, size=512, cropping='random', crop_scale=[1, 1]):
    if cropping in ['random', 'random_long_edge']:
        h, w, = img.shape[0], img.shape[1]
        crop = min(h, w)
        crop = int(torch.empty(1).uniform_(crop_scale[0], crop_scale[1]).item() * crop)
        offset_h = np.random.randint(0, h - crop + 1)
        offset_w = np.random.randint(0, w - crop + 1)
        img = img[offset_h:offset_h + crop, offset_w:offset_w + crop]
    elif cropping in ['center', 'center_long_edge']:
        h, w, = img.shape[0], img.shape[1]
        crop = min(h, w)
        crop = int(torch.empty(1).uniform_(crop_scale[0], crop_scale[1]).item() * crop)
        img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    elif cropping == 'crop':
        h, w, = img.shape[0], img.shape[1]
        crop = min(h, w, size)
        crop = int(torch.empty(1).uniform_(crop_scale[0], crop_scale[1]).item() * crop)
        offset_h = np.random.randint(0, h - crop + 1)
        offset_w = np.random.randint(0, w - crop + 1)
        img = img[offset_h:offset_h + crop, offset_w:offset_w + crop]
    else:
        raise NotImplementedError
    return img


class PersonalizedMulti(Dataset):
    def __init__(
        self,
        data_root="",
        size=None,
        repeats=100,
        interpolation="lanczos",
        flip_p=0.5,
        which_set="train",
        placeholder_token="sks",
        per_image_tokens=False,
        cropping='random_long_edge',
        crop_scale=[1, 1],
        mixing_prob=0.25,
        coarse_class_text=None,
        reg=False,
        use_small_template=False,
        delimiters = ",|:|;",
        **kwargs,
    ):
        # NOTE: split str to list
        
        data_root = re.split(delimiters, data_root)
        placeholder_token = re.split(delimiters, placeholder_token)
        # import ipdb
        # ipdb.set_trace()
        if coarse_class_text:
            coarse_class_text = re.split(delimiters, coarse_class_text) 
            coarse_class_text = [(None if (s in ['none', 'None', 'null', 'Null']) else s) for s in coarse_class_text]
        else:
            coarse_class_text = [None] * len(data_root)
        assert len(placeholder_token) == len(data_root) == len(coarse_class_text)
        self.keys = placeholder_token
        self.data_root = {k: v for k, v in zip(self.keys, data_root)}
        self.placeholder_token = {k: v for k, v in zip(self.keys, placeholder_token)}
        self.coarse_class_text = {k: v for k, v in zip(self.keys, coarse_class_text)}
        
        self.image_paths = {k: [os.path.join(self.data_root[k], file_path) for file_path in os.listdir(self.data_root[k])] if os.path.isdir(self.data_root[k])else [self.data_root[k]] for k in self.keys}

        self.num_images = {k: len(self.image_paths[k]) for k in self.keys}
        self._length = max([self.num_images[k] for k in self.keys])
        if which_set == "train":
            self._length = self._length * repeats

        self.reg = reg
        # self.per_image_tokens = per_image_tokens
        self.cropping = cropping
        self.crop_scale = crop_scale
        self.mixing_prob = mixing_prob
        
        self.use_small_template = use_small_template
        self.templates = {
            k: self.setup_templates(
                placeholder_token=self.placeholder_token[k],
                coarse_class_text=self.coarse_class_text[k],
                reg=reg, use_small_template=use_small_template,
            ) for k in self.keys
        }

        self.size = size
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def setup_templates(self, placeholder_token='sks', coarse_class_text='dog', reg=False, use_small_template=False):
        if reg:  # NOTE: reg dataset
            if coarse_class_text:
                placeholder_string = f"{coarse_class_text}"
                templates = imagenet_templates_small if use_small_template else reg_templates_smallest
                templates = [t.format(placeholder_string) for t in templates]
            else:
                templates = reg_templates_no_class_small if use_small_template else reg_templates_no_class_smallest
        else:  # NOTE: train dataset
            if coarse_class_text:
                placeholder_string = f"{placeholder_token} {coarse_class_text}"
            else:
                placeholder_string = f"{placeholder_token}"
            templates = imagenet_templates_small if use_small_template else reg_templates_smallest
            templates = [t.format(placeholder_string) for t in templates]
        return templates

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        key = random.choice(self.keys)

        example = {}

        image = Image.open(self.image_paths[key][i % self.num_images[key]])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        img = crop_image(img, size=self.size, cropping=self.cropping, crop_scale=self.crop_scale)
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        text = random.choice(self.templates[key])

        example["caption"] = text.rstrip()
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["label"] = [self.keys.index(key)]  
        # import ipdb
        # ipdb.set_trace()
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
        reg = False
    ):
        self.data_root = data_root
        assert os.path.isfile(self.data_root), f"SinImageDataset requires a path to a image file, not a directory. Got {self.data_root}."

        self.image_paths = [self.data_root]*100

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

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
        assert os.path.isfile(self.data_root), f"SinImageDataset requires a path to a image file, not a directory. Got {self.data_root}."

        self.num_images = 100
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

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
            
        self.image = image.resize((self.high_resolution, self.high_resolution), resample=self.interpolation)
        
    def __len__(self):
        return self._length
    
    def _random_crop(self, pil_image):
        patch_size_y = int(
            (self.high_resolution//self.latent_scale) * (random.random() * (self.max_crop_frac - self.min_crop_frac) + self.min_crop_frac))
        patch_size_x = int(
            (self.high_resolution//self.latent_scale) * (random.random() * (self.max_crop_frac - self.min_crop_frac) + self.min_crop_frac))
        crop_y = random.randrange((self.high_resolution//self.latent_scale) - patch_size_y + 1)
        crop_x = random.randrange((self.high_resolution//self.latent_scale) - patch_size_x + 1)
        return pil_image.crop((crop_x * self.latent_scale, crop_y * self.latent_scale, (crop_x + patch_size_x) * self.latent_scale, (crop_y + patch_size_y) * self.latent_scale)).resize(
            (self.size, self.size),
            resample=self.interpolation), crop_y, crop_x, crop_y + patch_size_y, crop_x + patch_size_x
    
    def __getitem__(self, i):
        example = {}
        image = deepcopy(self.image)


        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"


        text = random.choice(training_templates_smallest).format(placeholder_string)

            
        example["caption"] = text

        if random.random() < self.rec_prob:
            image, crop_y, crop_x, crop_y1, crop_x1 = self._random_crop(image)
            crop_area = torch.tensor([crop_y, crop_x, crop_y1, crop_x1])
        else:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            crop_area = torch.tensor([0, 0, self.high_resolution//self.latent_scale, self.high_resolution//self.latent_scale])
            
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example['crop_boxes'] = crop_area
        return example