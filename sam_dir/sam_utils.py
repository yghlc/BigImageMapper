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

# this would import the global environment, not the "datasets" in the local folder
# there is also another "Dataset" used in torch.utils.data, so rename it to huggingface_Dataset
from datasets import Dataset as huggingface_Dataset
from PIL import Image

# Save the original sys.path
original_path = sys.path[:]

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as  io_function
import parameters

# import from an upper level of folder, avoid the conflit with "datasets" from hugging face
code_dir2 = os.path.abspath(os.path.join(code_dir, '..'))
sys.path.insert(0, code_dir2)
# import Landuse_DL.datasets.vector_gpd as vector_gpd
import Landuse_DL.datasets.raster_io as raster_io

# Restore the original sys.path
sys.path = original_path

def get_model_type_hf(model_type):
    # get the pre-trained model string on hugging face
    if model_type == 'vit_b':
        pre_str = "facebook/sam-vit-base"
    elif model_type == 'vit_l':
        pre_str = "facebook/sam-vit-large"
    elif model_type == 'vit_h':
        pre_str = "facebook/sam-vit-huge"
    else:
        raise ValueError('Unknown mmodel type: %s'%model_type)

    return pre_str

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
    # print('debuging',idx)
    # print('debuging',item)
    image = item["image"]
    # print(image)
    # print(item["label"])
    # TODO: multiply it by 255?, because we set the ground truth as 1, but SAM may use 255?
    ground_truth_mask = np.array(item["label"])
    # print(ground_truth_mask.shape)

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
    #TODO: need to assigned this from para_file
    target_size = (256, 256)  # Desired target size for the images

    # read training images
    img_list = [ os.path.join('split_images', img_id.strip() + img_ext)  for img_id in  img_ids ]
    mask_list = [ os.path.join('split_labels', img_id.strip() + img_ext) for img_id in  img_ids ]
    # read image file to Pillow images and store them in a dictionary
    # dataset_dict = {
    #     "image": [Image.open(img).resize(target_size) for img in img_list],
    #     "label": [Image.open(mask).resize(target_size) for mask in mask_list],
    # }
    dataset_dict = {
        "image": [Image.open(img).crop((0,0,256,256)) for img in img_list],
        "label": [Image.open(mask).crop((0,0,256,256)) for mask in mask_list],
    }
    # Create a list to store the indices of non-empty masks
    valid_indices = [i for i, mask in enumerate(dataset_dict['label']) if mask.max() != 0]
    # Filter the image and mask arrays to keep only the non-empty pairs
    dataset_dict = {
        "image": dataset_dict['image'][valid_indices],
        "label": dataset_dict['label'][valid_indices],
    }

    print('reading %d image patches into memory, e.g,'%len(img_list), 'size of the first one:', dataset_dict['image'][0].size, dataset_dict['label'][0].size )
    for img, labl in zip(dataset_dict['image'],dataset_dict['label']):
        if img.size != target_size or labl.size != target_size:
            print(img.size)
            print(labl.size)
            raise ValueError('size different')
    # Create the dataset using the datasets.Dataset class
    dataset = huggingface_Dataset.from_dict(dataset_dict)
    # print('debuging', dataset)
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


def test_prepare_dataset_for_SAM_RS():
    import matplotlib.pyplot as plt
    para_file = 'main_para.ini'
    dataset, valid_dataset = prepare_dataset_for_SAM_RS(para_file)

    # Initialize the processor
    from transformers import SamProcessor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    train_dataset = SAM_RS_Dataset(dataset=dataset, processor=processor)
    example = train_dataset[0]
    # print(example)
    for k, v in example.items():
        print(k, v.shape)
        print(v)

    batch_size = 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    batch = next(iter(train_dataloader))
    for k, v in batch.items():
        print(k, v.shape)

    # img_num = random.randint(0, filtered_images.shape[0] - 1)
    img_num = 1
    example_image = dataset[img_num]["image"]
    example_mask = dataset[img_num]["label"]

    print(np.array(example_image).shape)
    print(np.array(example_mask).shape)
    print(np.unique(np.array(example_mask), return_counts=True))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first image on the left
    # axes[0].imshow(np.array(example_image), cmap='gray')  # Assuming the first image is grayscale
    axes[0].imshow(np.array(example_image))
    axes[0].set_title("Image")

    # Plot the second image on the right
    axes[1].imshow(example_mask, cmap='gray')  # Assuming the second image is grayscale
    axes[1].set_title("Mask")

    # Hide axis ticks and labels
    # for ax in axes:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])

    # Display the images side by side
    plt.show()

def main():
    pass

if __name__ == '__main__':
    test_prepare_dataset_for_SAM_RS()
    pass