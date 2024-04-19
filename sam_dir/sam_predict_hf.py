#!/usr/bin/env python
# Filename: sam_predict_hf.py 
"""
introduction: run SAM prediction, using the function from hugging face

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 April, 2024
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser

from transformers import SamModel, SamConfig, SamProcessor
import torch

from PIL import Image
import numpy as np


# Save the original sys.path
original_path = sys.path[:]
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
# Restore the original sys.path
sys.path = original_path

from sam_utils import get_model_type_hf, get_bounding_box

# this would import the global environment, not the "datasets" in the local folder
# there is also another "Dataset" used in torch.utils.data, so rename it to huggingface_Dataset
from datasets import Dataset as huggingface_Dataset

# must be before importing matplotlib.pyplot or pylab!
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sam_prediction_hg_small_img(trained_model, dataset, save_dir='./'):
    # run prediction on small images (not big remote sensing imagery)
    pass


def sam_inference_main(para_file,train_model = None):
    print(datetime.now(), "running Segment Anything models using hugging face API and trained models")

    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')

    # checkpoint = parameters.get_file_path_parameters(network_ini, 'checkpoint')
    model_type = parameters.get_string_parameters(network_ini, 'model_type')
    batch_size = parameters.get_digit_parameters(network_ini, 'batch_size', 'int')

    # Load the model configuration
    model_config = SamConfig.from_pretrained(get_model_type_hf(model_type))
    processor = SamProcessor.from_pretrained(get_model_type_hf(model_type))

    # Create an instance of the model architecture with the loaded configuration
    trained_model = SamModel(config=model_config)

    # Update the model by loading the weights from saved file.
    if train_model is not None:
        trained_model.load_state_dict(torch.load(train_model))

    # set the device to cuda if available, otherwise use cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model.to(device)

    # test prediction on small images
    test_am_prediction_hg_small_img(trained_model, processor, device)


def test_am_prediction_hg_small_img(trained_model, processor, device):
    # run prediction on small images (not big remote sensing imagery)
    para_file = 'main_para.ini'
    img_list_txt = 'list/val_list.txt'
    img_ids = [item.strip() for item in io_function.read_list_from_txt(img_list_txt)]
    img_ext = parameters.get_string_parameters_None_if_absence(para_file, 'split_image_format')

    target_size = (256, 256)  # Desired target size for the images

    # read training images
    img_list = [os.path.join('split_images', img_id.strip() + img_ext) for img_id in img_ids]
    mask_list = [os.path.join('split_labels', img_id.strip() + img_ext) for img_id in img_ids]
    # read image file to Pillow images and store them in a dictionary
    # dataset_dict = {
    #     "image": [Image.open(img) for img in img_list],         # .resize(target_size)
    #     "label": [Image.open(mask) for mask in mask_list],      # .resize(target_size)
    # }
    dataset_dict = {
        "image": [Image.open(img).resize(target_size) for img in img_list],
        "label": [Image.open(mask).resize(target_size) for mask in mask_list],
    }
    print('reading %d image patches into memory, e.g,' % len(img_list), 'size of the first one:',
          dataset_dict['image'][0].size, dataset_dict['label'][0].size)

    # Create the dataset using the datasets.Dataset class
    dataset = huggingface_Dataset.from_dict(dataset_dict)

    test_data = dataset[0]
    print(img_list[0])
    test_image = test_data['image']

    # get box prompt based on ground truth segmentation map
    ground_truth_mask = np.array(test_data["label"])
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image + box prompt for the model
    # Expected either PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray
    inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

    # Move the input tensor to the GPU if it's not already there
    inputs = {k: v.to(device) for k, v in inputs.items()}

    trained_model.eval()

    # forward pass
    with torch.no_grad():
        outputs = trained_model(**inputs, multimask_output=False)

    print('outputs', outputs.shape)
    print(np.max(outputs), np.min(outputs), np.mean(outputs))

    # apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    print('medsam_seg_prob',medsam_seg_prob.shape)
    print(np.max(medsam_seg_prob), np.min(medsam_seg_prob), np.mean(medsam_seg_prob) )
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    print('medsam_seg',medsam_seg.shape)
    print(np.max(medsam_seg), np.min(medsam_seg), np.mean(medsam_seg), np.unique(medsam_seg, return_counts=True))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the first image on the left
    axes[0].imshow(np.array(test_image), cmap='gray')  # Assuming the first image is grayscale
    axes[0].set_title("Image")

    # Plot the second image on the right
    axes[1].imshow(medsam_seg, cmap='gray')  # Assuming the second image is grayscale
    axes[1].set_title("Mask")

    # Plot the second image on the right
    axes[2].imshow(medsam_seg_prob)  # Assuming the second image is grayscale
    axes[2].set_title("Probability Map")

    # # Hide axis ticks and labels
    # for ax in axes:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])

    # Display the images side by side
    # plt.show()
    plt.savefig('output.jpg')




def main(options, args):
    # test_am_prediction_hg_small_img("")

    para_file = args[0]
    train_model = options.trained_model
    sam_inference_main(para_file)

if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2023-06-15")
    parser.description = 'Introduction: run segmentation using segment anything model '

    parser.add_option("-m", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model for prediction")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
