#!/usr/bin/env python
# Filename: cluster_analysis.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 July, 2025
"""

import os,sys
from optparse import OptionParser

import torch
import skimage.io as io
import PIL.Image

from model_visualization import load_clip_model


code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
# import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function


def get_image_path_list(input, file_extension='.tif'):
    image_path_list = []
    if input.endswith(".txt"):
        # read file name from the txt file
        image_path_list = io_function.read_list_from_txt(input)
    elif os.path.isfile(input):
        image_path_list.append(image_path_list)
    elif os.path.isdir(input):
        # read files name from a folder
        image_path_list = io_function.get_file_list_by_pattern(image_path_list, f'*{file_extension}')
    else:
        raise IOError(f'Cannot recognize the input: {image_path_list}')

    return image_path_list

def get_img_features(image_path_list, model, preprocess, device):
    """obtain the feature (vector) of images in latent space"""

    feature_list = []

    with torch.no_grad():
        for idx, image_path in enumerate(image_path_list):
            print(f' {idx + 1}/{len(image_path_list)}, generating feature for {os.path.basename(image_path)}')
            image = io.imread(image_path)
            pil_image = PIL.Image.fromarray(image)
            image = preprocess(pil_image).unsqueeze(0).to(device)

            image_feature = model.encode_image(image).to(device, dtype=torch.float32)
            feature_list.append(image_feature)

    return feature_list



def main(options, args):

    image_list = get_image_path_list(args[0])
    trained_model = options.trained_model
    model_type = options.model_type

    device = "cuda"  if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = load_clip_model(device, model_type=model_type, trained_model=trained_model)

    image_features = get_img_features(image_list, clip_model,preprocess, device)

    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] image_path or image_folder or image_list.txt "
    parser = OptionParser(usage=usage, version="1.0 2025-07-3")
    parser.description = 'Introduction: cluster analysis of small images using AI foundation models '

    parser.add_option("-s", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model")

    parser.add_option("-t", "--model_type",
                      action="store", dest="model_type", default='ViT-B/32',
                      help="the model type or architecture")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)