#!/usr/bin/env python
# Filename: class_utils.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 27 January, 2024
"""



import os,sys
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as  io_function

class RSVectorDataset(Dataset):
    pass

class RSPatchDataset(Dataset):
    def __init__(self, image_path_list, image_labels, label_txt = 'label.txt', transform=None, test=False):
        self.img_list = image_path_list
        self.labels = image_labels

        label_list = [[item.split(',')[0], int(item.split(',')[1])] for item in io_function.read_list_from_txt(label_txt)]
        # arr_t = np.array(label_list).T
        label_list = np.array(label_list).T.tolist()    # switch the row and column
        self.transform = transform
        self.test = test

        self.classes = label_list[0]

    def __getitem__(self, index):
        im_path = self.img_list[index]
        label = self.labels[index]
        # im_path = os.path.join(self.data_root, 'Images', im_path0)
        im = Image.open(im_path)
        if self.transform is not None:
            im = self.transform(im)

        return im, label, im_path

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    pass