#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Introduction:

Author: Huang Lingcao
Email: huanglingcao@gmail.com
Created: 2026-03-23
"""
import os,sys

from typing import ClassVar

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as  io_function
import basic_src.basic as basic
import datasets.vector_gpd as vector_gpd
import parameters

from img_classification.class_utils import get_training_data_dir, get_merged_training_data_txt

# import terratorch
import numpy as np
from PIL import Image

from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import NonGeoClassificationDataset

from torchgeo.datamodules import UCMercedDataModule
from torchgeo.datasets import UCMerced

from torchvision import transforms
import torch

## define RSPatchTxtDataset and RSPatchTxtModule for training with custom data, which is organized in txt file with image path and label, e.g.,
# /path/to/image1.jpg 0
# /path/to/image2.jpg 1

# in another txt file, we can define the class labels, e.g.,
# others, 0
# thawslump, 1

class RSPatchTxtDataset(NonGeoClassificationDataset):
    '''
    A dataset class for training with custom data, which is organized in txt file with image path and label, e.g.,
    /path/to/image1.jpg 0
    /path/to/image2.jpg 1

    In another txt file, we can define the class labels, e.g.,
    others, 0
    thawslump, 1
    '''

    splits = ('train', 'val', 'test')
    split_filenames: ClassVar[dict[str, str | None]] = {
        'train': None,
        'val': None,
        'test': None,
    }

    def __init__(self, train_txt=None, valid_txt=None, test_txt=None, label_txt='label.txt', transforms=None, split: str = 'train'):
        assert split in self.splits

        split_files = {
            'train': train_txt,
            'val': valid_txt,
            'test': test_txt,
        }
        split_txt = split_files[split]
        if split_txt is None:
            raise ValueError('The txt file for split %s is not set' % split)

        self.img_list = []
        self.labels = []

        # # tmp = item.rsplit(',', 1)  # Split into two parts using the last comma
        # label_list = [[item.rsplit(',', 1)[0], int(item.rsplit(',', 1)[1])] for item in io_function.read_list_from_txt(label_txt)]
        # # arr_t = np.array(label_list).T
        # label_list = np.array(label_list).T.tolist()    # switch the row and column
        # self.classes = label_list[0]

        label_items = []
        for item in io_function.read_list_from_txt(label_txt):
            name, idx = item.rsplit(',', 1)
            label_items.append((int(idx), name.strip()))

        label_items = sorted(label_items, key=lambda x: x[0])
        self.classes = [name for _, name in label_items]

        # print(f'classes: {self.classes}')

        self.transforms = transforms

        with open(split_txt) as f:
            for fn in f:
                line = fn.strip()
                if len(line) < 1:
                    continue
                # Split from the right once to preserve image paths with spaces.
                img_path, label = line.rsplit(maxsplit=1)
                self.img_list.append(img_path)
                label_int = int(label)
                if label_int < 0 or label_int >= len(self.classes):
                    raise ValueError(f'Label index {label_int} is out of bounds for classes: {self.classes}')
                self.labels.append(int(label))


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        im_path = self.img_list[index]
        label = self.labels[index]


        # im_path = os.path.join(self.data_root, 'Images', im_path0)
        im = Image.open(im_path).convert('RGB')

        # to check? (this is from torchgeo.datasets.geo.NonGeoClassificationDataset._load_image)
        # img, label = ImageFolder.__getitem__(self, index)
        # array: np.typing.NDArray[np.int_] = np.array(img)
        # tensor = torch.from_numpy(array).float()
        # # Convert from HxWxC to CxHxW
        # tensor = tensor.permute((2, 0, 1))
        # label = torch.tensor(label)

        if self.transforms is not None:
            im = self.transforms(im)
        else:
            im = transforms.ToTensor()(im)

        samples = {'image': im, 'label': torch.tensor(label), 'filename': im_path}

        return samples
    
    def plot(self, sample):
        import matplotlib.pyplot as plt
        im = sample['image']
        label = sample['label']
        filename = sample['filename']
        # Convert from CxHxW to HxWxC for plotting.
        im = im.permute((1, 2, 0)).numpy()
        plt.imshow(im)
        plt.title(f'Label: {self.classes[label]}')
        plt.xlabel(filename)
        plt.axis('off')
        # plt.show()


class RSPatchTxtModule(NonGeoDataModule):
    '''
    A datamodule class for training with custom data, which is organized in txt file with image path and label, e.g.,
    /path/to/image1.jpg 0
    /path/to/image2.jpg 1

    In another txt file, we can define the class labels, e.g.,
    others, 0
    thawslump, 1
    '''

    def __init__(
        self,
        train_txt,
        valid_txt,
        label_txt,
        test_txt=None,
        batch_size=32,
        num_workers=4,
        transforms=None,
        **kwargs,
    ):
        super().__init__(
            RSPatchTxtDataset,
            batch_size,
            num_workers,
            train_txt=train_txt,
            valid_txt=valid_txt,
            test_txt=test_txt,
            label_txt=label_txt,
            transforms=transforms,
            **kwargs,
        )

    
    
def get_data_transforms(be_normalzed=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # simple data resize and normalization for both training and validtion. (no data augmentation) 
    # this may crop some info at the edge, but avoid different image size. 

    if be_normalzed:
        data_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)
                                             ])
    else:
        # just convert to tensor, but not normalized to mean/std,
        data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         ])

    return data_transform



def prepare_train_val_txt(WORK_DIR, para_file):

    training_regions = parameters.get_string_list_parameters_None_if_absence(para_file,'training_regions')
    if training_regions is None or len(training_regions) < 1:
        raise ValueError('No training area is set in %s'%para_file)

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    training_data_dir = get_training_data_dir(WORK_DIR)
    merged_training_data_txt = get_merged_training_data_txt(training_data_dir, expr_name,len(training_regions))
    merged_training_data_txt_all = io_function.get_name_by_adding_tail(merged_training_data_txt,'all')
    merged_training_data_txt_notSel = io_function.get_name_by_adding_tail(merged_training_data_txt,'notSel')
    merged_training_data_txt_valid = io_function.get_name_by_adding_tail(merged_training_data_txt,'valid')

    train_txt = None
    valid_txt = None

    if os.path.isfile(merged_training_data_txt):
        train_txt = merged_training_data_txt
    else:
        train_txt = None
        basic.outputlogMessage('Please run img_classification/get_organize_training_data.py first to prepare and organize the training data')
        return None, None

    # use *valid.txt for validation, if does not exist, use *notSel.txt, else, use all.txt
    if os.path.isfile(merged_training_data_txt_valid):
        valid_txt = merged_training_data_txt_valid
    elif os.path.isfile(merged_training_data_txt_notSel):
        valid_txt = merged_training_data_txt_notSel
    else:
        valid_txt = merged_training_data_txt_all

    return train_txt, valid_txt


def organize_training_data_from_txt(para_file, train_data_txt, save_dir, split='train'):
    '''
    read and create training data from a txt file
    :param para_file:
    :param train_data_txt:
    :param save_dir:
    :return:
    '''

    class_labels = parameters.get_file_path_parameters(para_file, 'class_labels')
    image_path_labels = [item.split() for item in io_function.read_list_from_txt(train_data_txt)]
    image_path_list = [item[0] for item in image_path_labels]  # it's already absolute path
    image_labels = [int(item[1]) for item in image_path_labels]

    label_list = [[item.rsplit(',', 1)[0], int(item.rsplit(',', 1)[1])] for item in io_function.read_list_from_txt(class_labels)]
    # arr_t = np.array(label_list).T
    digit_classname_s = np.array(label_list).T.tolist()    # switch the row and column
    # print('debuging:',digit_classname_s)

    # copy the training data to the save_dir, and re-organize the data into class-based sub-folders
    for img_path, label in zip(image_path_list, image_labels):
        class_name = digit_classname_s[0][label]
        print('debuging:', img_path, label, class_name)
        class_dir = os.path.join(save_dir, split, class_name)
        if os.path.isdir(class_dir) is False:
            io_function.mkdir(class_dir)
        save_path = os.path.join(class_dir, os.path.basename(img_path))
        io_function.copy_file_to_dst(img_path, save_path,overwrite=False)

    return True

def prepare_train_val_data_folder(WORK_DIR, para_file):

    # orginize the training and validation data into folders: 
    # data_root/
    # train/
    #   class0/
    #   class1/
    # val/
    #   class0/
    #   class1/
    # test/
    #   class0/
    #   class1/

    training_regions = parameters.get_string_list_parameters_None_if_absence(para_file,'training_regions')
    if training_regions is None or len(training_regions) < 1:
        raise ValueError('No training area is set in %s'%para_file)

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    training_data_dir = get_training_data_dir(WORK_DIR)
    merged_training_data_txt = get_merged_training_data_txt(training_data_dir, expr_name,len(training_regions))
    merged_training_data_txt_all = io_function.get_name_by_adding_tail(merged_training_data_txt,'all')
    merged_training_data_txt_notSel = io_function.get_name_by_adding_tail(merged_training_data_txt,'notSel')
    merged_training_data_txt_valid = io_function.get_name_by_adding_tail(merged_training_data_txt,'valid')
    valid_dataset = None

    # re-organize the data train, val sets for training. 
    if os.path.isfile(merged_training_data_txt):
        # in_dataset = create_training_data_from_txt(para_file,merged_training_data_txt,transform,test=test)
        organize_training_data_from_txt(para_file, merged_training_data_txt, training_data_dir, split='train')
    else:
        basic.outputlogMessage('Please run img_classification/get_organize_training_data.py first to prepare and organize the training data')
        return False

    # use *valid.txt for validation, if does not exist, use *notSel.txt, else, use all.txt
    if os.path.isfile(merged_training_data_txt_valid):
        organize_training_data_from_txt(para_file, merged_training_data_txt_valid, training_data_dir, split='val')
    elif os.path.isfile(merged_training_data_txt_notSel):
        organize_training_data_from_txt(para_file, merged_training_data_txt_notSel, training_data_dir, split='val')
    else:
        organize_training_data_from_txt(para_file, merged_training_data_txt_all, training_data_dir, split='val')

    return training_data_dir



def main():
    pass


if __name__ == "__main__":
    main()
