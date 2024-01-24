#!/usr/bin/env python
# Filename: zeroshort_classify_clip.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 January, 2024
"""



import os,sys
import os.path as osp
from optparse import OptionParser
from datetime import datetime
import time
import GPUtil

from PIL import Image

import numpy as np
import torch
import clip

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function

def test_classification_ucm(model, preprocess):
    data_dir = os.path.expanduser('~/Data/image_classification/UCMerced_LandUse')


    # read classes info
    label_list_txt = os.path.join(data_dir,'label_list.txt')
    class_labels = [item.split(',')[0] for item in io_function.read_list_from_txt(label_list_txt) ]
    text_descriptions = [f"This is a satellite image of a {label}" for label in class_labels]
    text_tokens = clip.tokenize(text_descriptions).cuda()

    # process text
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)


    # randomly read ten images
    image_txt = os.path.join(data_dir,'all.txt')
    image_list = [ item.split() for item in io_function.read_list_from_txt(image_txt)]
    image_path_list = [ os.path.join(data_dir,'Images', item[0]) for item in image_list]
    image_class_list = [ int(item[1]) for item in image_list]

    images = []
    # sel_index = [0, 10, 100, 200, 300, 500, 700, 900, 1000,1500, 2000]
    sel_index = [item for item in range(len(image_path_list))]
    for idx in sel_index:
        image = Image.open(image_path_list[idx]).convert("RGB")
        images.append(preprocess(image))
    image_input = torch.tensor(np.stack(images)).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs_5, top_labels_5 = text_probs.cpu().topk(5, dim=-1)
    print(top_probs_5)
    print(top_labels_5)

    top_probs_1, top_labels_1 = text_probs.cpu().topk(1, dim=-1)
    print(top_probs_1)
    print(top_labels_1)

    # output accuracy
    # top1 accuray
    top_labels_1 = top_labels_1.numpy().squeeze()
    hit_count = 0
    for idx, pred_l in zip(sel_index,top_labels_1):
        if image_class_list[idx] == pred_l:
            hit_count += 1
    print('top 1 accuray: (%d /%d): %f'%(hit_count, len(sel_index), 100.0*hit_count/len(sel_index) ))

    # top5 accuray 
    top_labels_5 = top_labels_5.numpy().squeeze()
    # print(top_labels_5)
    hit_count = 0
    for idx, pred_l_s in zip(sel_index,top_labels_5):
        # print(pred_l_s)
        if image_class_list[idx] in pred_l_s:
            hit_count += 1
    print('top 5 accuray: (%d /%d): %f'%(hit_count, len(sel_index), 100.0*hit_count/len(sel_index) ))
    




def main(options, args):

    para_file = args[0]
    trained_model = options.trained_model

    model, preprocess = clip.load("ViT-B/32")
    # model, preprocess = clip.load("ViT-L/14")
    model.cuda().eval() # to download the pre-train models.

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    test_classification_ucm(model,preprocess)

    # mmseg_parallel_predict_main(para_file,trained_model)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-01-22")
    parser.description = 'Introduction: run prediction in parallel on using clip '

    parser.add_option("-m", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model for prediction")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)