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
import numpy as np

import cv2
# this would import the global environment, not the "datasets" in the local folder
from datasets import Dataset
from PIL import Image


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
        return image, image_path, label_raster.float()


class SAM_RS_Dataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  processor is from hugging face transformers
  it will also handle RS image patches

  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs


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

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

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

def read_one_dataset_PIL(img_list_txt, img_ext):

    img_ids = [item.strip() for item in io_function.read_list_from_txt(img_list_txt)]

    # read training images
    img_list = [ os.path.join('split_images', img_id.strip() + img_ext)  for img_id in  img_ids ]
    mask_list = [ os.path.join('split_labels', img_id.strip() + img_ext) for img_id in  img_ids ]
    # read image file to Pillow images and store them in a dictionary
    dataset_dict = {
        "image": [Image.open(img) for img in img_list],
        "label": [Image.open(mask) for mask in mask_list],
    }
    print('reading %d image patches into memory, e.g,'%len(img_list), 'shape of the first one:', dataset_dict['image'][0].shape )
    # Create the dataset using the datasets.Dataset class
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def prepare_dataset_for_SAM_RS(para_file):

    training_list_txt = parameters.get_string_parameters(para_file, 'training_sample_list_txt')
    training_list_txt = os.path.join('list', training_list_txt)
    valid_list_txt = parameters.get_string_parameters(para_file, 'validation_sample_list_txt')
    valid_list_txt = os.path.join('list', valid_list_txt)

    # network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    # batch_size = parameters.get_digit_parameters(network_ini, 'batch_size', 'int')
    # process_num = parameters.get_digit_parameters(para_file, 'process_num', 'int')
    img_ext = parameters.get_string_parameters_None_if_absence(para_file, 'split_image_format')

    # read and create datasets.Dataset class for training
    training_dataset = read_one_dataset_PIL(training_list_txt,img_ext)

    # validation images
    valid_dataset = read_one_dataset_PIL(valid_list_txt, img_ext)
    return training_dataset, valid_dataset



def main():
    pass

if __name__ == '__main__':
    pass