# encoding: utf-8
"""
@author: ccj
@contact:
"""

import os
import pdb
import cv2

import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self,
                 img_id,
                 image_dir,
                 target,
                 transform=None
                 ):
        super(TrainDataset, self).__init__()
        self.img_id = img_id
        self.image_dir = image_dir
        self.target = target
        self.transform = transform

        assert (os.path.exists(image_dir)), "image folder doesn't exist"

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        # pdb.set_trace()
        img_id = self.img_id[idx]
        imgfile = f"{self.image_dir}/{img_id}.jpg"
        try:
            image = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        except OSError:
            print("image file doesn't exist")

        images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = self.target[idx]

        if self.transform is not None:
            images = self.transform(image=images)["image"]

        return {
            "image": images,
            "target": target
        }

