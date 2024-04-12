#!/usr/bin/env python
# Filename: sam_utils.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 April, 2024
"""

import os,sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from segment_anything.utils.transforms import ResizeLongestSide
import torch

import cv2

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as  io_function
import datasets.vector_gpd as vector_gpd
import datasets.raster_io as raster_io
import parameters

class RSPatchDataset(Dataset):
    """
    A PyTorch Dataset to load data from a current folder.

    Attributes
    ----------
    data_dir : str
        the root directory containing the images and annotations
    img_list_txt : str
        name of the txt file containing the annotations (in root_dir)
    para_file: str
        path to the main parameters
    transform : callable
        a function/transform to apply to each image

    Methods
    -------
    __getitem__(idx)
        returns the image, image path, and masks for the given index

    """

    def __init__(self, data_dir, para_file, img_list_txt, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_ids = io_function.read_list_from_txt(img_list_txt)
        self.para_file = para_file
        self.img_ext = parameters.get_string_parameters_None_if_absence(para_file, 'split_image_format')

    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_path = os.path.join('split_images', image_id.strip() + self.img_ext)
        label_path = os.path.join('split_labels', image_id.strip() + self.img_ext)

        # image_cv = cv2.imread(image_path)
        # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        image, nodata = raster_io.read_raster_all_bands_np(image_path)
        image = image.transpose(1, 2, 0)  # to opencv format  # in HWC uint8 format
        label_raster, nodata_l = raster_io.read_raster_all_bands_np(label_path)
        label_raster = label_raster.transpose(1, 2, 0)  # # to opencv format  # in HWC

        if self.transform:
            # image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
            image, label_raster = self.transform(image, label_raster)

        # bboxes = np.stack(bboxes, axis=0)
        # masks = np.stack(masks, axis=0)
        return image, image_path, torch.tensor(label_raster).float()


def get_totalmask(masks):
    """get all masks in to one image
    ARGS:
        masks (List[Tensor]): list of masks
    RETURNS:
        total_gt (Tensor): all masks in one image

    """
    total_gt = torch.zeros_like(masks[0][0,:,:])
    for k in range(len(masks[0])):
        total_gt += masks[0][k,:,:]
    return total_gt


class ResizeAndPad:
    """
    Resize and pad images and label_raster to a target size (for semantic segmentation).

    ...
    Attributes
    ----------
    target_size : int
        the target size of the image
    transform : ResizeLongestSide
        a transform to resize the image and masks
    """

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, label_raster):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        label_raster = self.transform.apply_image(label_raster)
        # masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)
        label_raster = self.to_tensor(label_raster)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        label_raster = transforms.Pad(padding)(label_raster)
        # masks = [transforms.Pad(padding)(mask) for mask in masks]

        # # Adjust bounding boxes
        # if bboxes is not None:
        #     bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        #     bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, label_raster #, bboxes


def load_datasets(para_file, img_size=1024):
    """ load the training and validation datasets in PyTorch DataLoader objects
    ARGS:
        img_size (Tuple(int, int)): image size, default = 1024
    RETURNS:
        train_dataloader (DataLoader): training dataset
        val_dataloader (DataLoader): validation dataset

    """

    transform = ResizeAndPad(img_size)
    cur_dir = os.getcwd()

    training_list_txt = parameters.get_string_parameters(para_file,'training_sample_list_txt')
    valid_list_txt = parameters.get_string_parameters(para_file,'validation_sample_list_txt')

    network_ini = parameters.get_string_parameters(para_file,'network_setting_ini')
    batch_size = parameters.get_digit_parameters(network_ini,'batch_size','int')

    process_num = parameters.get_digit_parameters(para_file,'process_num','int')

    traindata = RSPatchDataset(cur_dir,
                               para_file,
                               os.path.join('list', training_list_txt),
                               transform=transform)

    valdata = RSPatchDataset(cur_dir,
                             para_file,
                             os.path.join('list', valid_list_txt),
                             transform=transform)

    train_dataloader = DataLoader(traindata,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=process_num)

    val_dataloader = DataLoader(valdata,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=process_num)
    return train_dataloader, val_dataloader




def main():
    pass

if __name__ == '__main__':
    pass