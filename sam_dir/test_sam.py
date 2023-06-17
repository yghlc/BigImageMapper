#!/usr/bin/env python
# Filename: test_sam.py 
"""
introduction: testing the de

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 14 June 2023
"""

import os,sys

import torch
import torchvision
import cv2

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

import sys
samDir=os.path.expanduser('~/codes/PycharmProjects/segment-anything-largeImage')
sys.path.append(samDir)
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = os.path.join(os.path.expanduser('~/Data/models_deeplearning/segment-anything/'), "sam_vit_h_4b8939.pth")
model_type = "vit_h"

device = 'cpu'  #"cuda"

dataDir = os.path.expanduser('~/Data/tmp_data/test_segmentAnything')
tif_path = os.path.join(dataDir,'willow_river_2019.tif')
png_path = os.path.join(dataDir,'willow_river_2019.PNG')


def everything_mode():
    image = cv2.imread(tif_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)

    print(len(masks))
    print(type(masks))
    print(masks[0])

    print(len(masks))
    print(masks[0].keys())

def segment_points():
    # segment targets with input points (1 for positives, 0 for negative
    pass


def main():
    everything_mode()

    pass

if __name__ == '__main__':
    main()