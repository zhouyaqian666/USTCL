#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList_label(ImageList):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path

class MultiViewList(Dataset):
    is_first = True
    def __init__(self, list_dir, views=12, transform=None, target_transform=None, mode='RGB'):
        self.imgs = []
        step =int(12 / views)
        r = [0, 1, 2, 3, 4, 5]
        for i in r:
            suffix = "{:0>3d}.txt".format(i+1)
            image_list_path = os.path.join(list_dir, suffix)
            image_list = make_dataset(open(image_list_path).readlines(), None)
            if len(image_list) == 0:
                raise(RuntimeError("Found 0 images in subfolders of:" + image_list_path+"\n"))
            self.imgs.append(image_list)
        self.transform=transform
        self.target_transform=target_transform
        if mode == 'RGB':
            self.loader=rgb_loader
        elif mode == 'L':
            self.loader=l_loader

    def __getitem__(self, index):
        imgs=[]
        target=None
        for i in range(len(self.imgs)):
            path, target=self.imgs[i][index]
            img=self.loader(path)
            if self.transform is not None:
                img=self.transform(img)
            if self.target_transform is not None:
                target=self.target_transform(target)
            imgs.append(img)
        imgs=np.stack(imgs, axis=0)
        imgs=torch.from_numpy(imgs)
        return imgs, target

    def __len__(self):
        if len(self.imgs) != 0:
            return len(self.imgs[0])
        return 0

class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

