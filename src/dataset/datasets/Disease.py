from skimage import io
import os
import numpy as np
import cv2
import hashlib
import torch
from itertools import repeat
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import json
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageOps
import random
import cv2
import math
from ..utils import cvtColor, preprocess_input, verify_image_label
from multiprocessing.pool import ThreadPool

def label_convert(name):
    if name == 'NR':
        label = 0
    elif name == 'VM':
        label = 1
    elif name == 'IH':
        label = 2

    return label

def convert_box(box, scale_w, scale_h):
    box[..., 2] = box[..., 2] - box[..., 0]
    box[..., 3] = box[..., 3] - box[..., 1]

    box[..., 0] = box[..., 0] / scale_w
    box[..., 1] = box[..., 1] / scale_h
    box[..., 2] = box[..., 2] / scale_w
    box[..., 3] = box[..., 3] / scale_h
    dw = box[..., 2] / 2
    dh = box[..., 3] / 2
    box[..., 0] = box[..., 0] + dw
    box[..., 1] = box[..., 1] + dh
    return box / 640

class Disease(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, phase, transform=None, opt=None):
        self.dir_dataset = '/home/dell/Project/IHNet/data'
        # Determenistic "random" shuffle of the maps:

        self.phase = phase

        with open(os.path.join(self.dir_dataset, 'output_file.json')) as file:
            self.datadir = json.load(file)
            print('OK')

        random.seed(42)
        random.shuffle(self.datadir)

        if phase == 'train':
            self.length = int(len(self.datadir) * 0.8)
            # self.length = int(len(self.datadir))
        elif phase == 'val':
            self.length = int(len(self.datadir) * 0.2)

        # self.transform = transform
        self.opt = opt
        self.input_shape = self.opt.image_size
        opt.num_classes = 3
        opt.in_chans = (3, 3)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        if self.phase == 'val':
            index = int(len(self.datadir) * 0.8) + index


        data_json = self.datadir[index]

        name = data_json['label']
        gray = data_json['gray']
        color = data_json['color']
        gray_coords = data_json['gray_coords']
        phlebolith_count = data_json['phlebolith_count']
        phlebolith_coords = data_json['phlebolith_coords']

        im_file_gray = os.path.join(self.dir_dataset, name, gray)
        image_gray = self.maxmin_norm(cv2.imread(im_file_gray)).transpose(2, 0, 1)
        c, w, h = image_gray.shape
        scale_w = w / 640
        scale_h = h / 640
        image_gray = F.interpolate(torch.from_numpy(image_gray).unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=True).squeeze(0)

        if color != None:
            im_file_color = os.path.join(self.dir_dataset, name, color)
            image_color = self.maxmin_norm(cv2.imread(im_file_color)).transpose(2, 0, 1)
            image_color = F.interpolate(torch.from_numpy(image_color).unsqueeze(0), size=(640, 640), mode='bilinear',
                                       align_corners=True).squeeze(0)
        classes = label_convert(name)

        if len(gray_coords) == 0:
            gray_coords = np.array([0, 0, 0, 0])
        else:
            gray_coords = np.array(gray_coords[0]).flatten()


        # plt.imshow(np.array(image_gray * 255).transpose(1, 2, 0).astype(np.uint8))
        # plt.show()
        # if self.phase == 'train':
        # random_ver = random.randint(0, 1)
        # random_hor = random.randint(0, 1)
        # """垂直翻转"""
        # if random_ver == 0:
        #     image_gray = torch.flip(image_gray, dims=[1])
        #     if len(phlebolith_coords) != 0:
        #         for i in range(len(phlebolith_coords)):
        #             phlebolith_coords[i] = np.array(phlebolith_coords[i]).flatten()
        #             phlebolith_coords[i][1] = 640 - h - phlebolith_coords[i][1]
        #     if color != None:
        #         image_color = torch.flip(image_gray, dims=[1])
        #
        # """水平翻转"""
        # if random_hor == 0:
        #     image_gray = torch.flip(image_gray, dims=[2])
        #     if len(phlebolith_coords) != 0:
        #         for i in range(len(phlebolith_coords)):
        #             phlebolith_coords[i] = np.array(phlebolith_coords[i]).flatten()
        #             phlebolith_coords[i][0] = 640 - w - phlebolith_coords[i][0]
        #     if color != None:
        #         image_color = torch.flip(image_gray, dims=[2])

        # plt.imshow(np.array(image_gray * 255).transpose(1,2,0).astype(np.uint8))
        # plt.show()
        phlebolith_label = {}
        if len(phlebolith_coords) == 0:
            # phlebolith_coords = np.array([0, 0, 0, 0])
            phlebolith_label.update(
                {
                    'im_file': im_file_gray,
                    'ori_shape': [640, 640],
                    'resized_shape': [640, 640],
                    'img': image_gray,
                    'ratio_pad': ((1.0, 1.0), (0, 0)),
                    'cls': [0],
                    # 'bboxes': np.array([convert_box(np.array([0, 0, 0, 0]), scale_w, scale_h)]),
                    'bboxes': np.array([convert_box(np.array([0, 0, w, h]), scale_w, scale_h)]),
                    'batch_idx': [gray],
                }
            )
        # np.empty([0, 4])np.array([np.array([0, 0, 0, 0])]),
        else:
            # phlebolith_coords = np.array(phlebolith_coords[0]).flatten()
            cls = []
            bboxed = []
            batch_idx = []
            for i in range(len(phlebolith_coords)):
                bboxed.append(convert_box(np.array(phlebolith_coords[i]).flatten(), scale_w, scale_h))
                cls.append(1)
                batch_idx.append(gray)

            phlebolith_label.update(
                {
                    'im_file': im_file_gray,
                    'ori_shape': [640, 640],
                    'resized_shape': [640, 640],
                    'img': image_gray,
                    'ratio_pad': ((1.0, 1.0), (0, 0)),
                    'cls': cls,
                    # 'bboxed': convert_box(np.array(bboxed)),
                    'bboxes': np.array(bboxed),
                    'batch_idx': batch_idx,
                }
            )

        random_int = random.randint(0, 1)
        # if random_int == 0 or color == None:
        #     image_color = torch.zeros_like(image_gray)
        if color == None:
            image_color = torch.zeros_like(image_gray)

        return image_gray, image_color, gray_coords, phlebolith_label, classes

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.length
        # return 50

    def maxmin_norm(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    # transforms = CustomDataAugmentation(256, 0.08)
    train_dataset = Luojiassr('val')
    # img_path = '/data/HyperSpectralNet/luojiassr/img256_train_new/13_obj_0_1__0_1_.tif'
    # label_path = '/data/HyperSpectralNet/luojiassr/label256_train_new/1_obj_0_0__0_0__0_1_.tif'
    # im_width, im_height, im_bands, im_proj, im_geotrans, im_data = train_dataset.read_img(img_path)
    # l_width, l_height, l_bands, l_proj, l_geotrans, l_data = dataset.read_img(label_path)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    new = torch.zeros(1, dtype=torch.int64)
    for im_data, l_data in train_loader:
        new = torch.cat([new, torch.unique(l_data)])
        new = torch.unique(new)

    print('OK')
    print(new)


