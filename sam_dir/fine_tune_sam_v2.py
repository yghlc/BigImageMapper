#!/usr/bin/env python
# Filename: fine_tune_sam_v2.py 
"""
introduction: fine tune sam models, based on some models from hugging face ransformers (https://github.com/huggingface/transformers)

Huggingface's version of SAM on https://huggingface.co/docs/transformers/en/model_doc/sam, look like easy to use

training of the SAM (based on hugging face's version):
https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb


authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 April, 2024
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser
import torch
import numpy as np
from tqdm import tqdm
from statistics import mean

# Save the original sys.path
original_path = sys.path[:]
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
# Restore the original sys.path
sys.path = original_path
#
from transformers import SamProcessor
from transformers import SamModel
from torch.optim import Adam
import monai

from torch.utils.data import DataLoader

from sam_utils import SAM_RS_Dataset, prepare_dataset_for_SAM_RS

def get_model_type_hf(model_type):
    # get the pre-trained model string on hugging face
    if model_type == 'vit_b':
        pre_str = "facebook/sam-vit-base"
    elif model_type == 'vit_l':
        pre_str = "facebook/sam-vit-large"
    elif model_type == 'vit_h':
        pre_str = "facebook/sam-vit-huge"
    else:
        raise ValueError('Unknow mmodel type: %s'%model_type)

    return pre_str

def fine_tune_sam(WORK_DIR, para_file, pre_train_model='', gpu_num=1,b_evaluate=True):
    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # checkpoint = parameters.get_file_path_parameters(network_ini, 'checkpoint')
    model_type = parameters.get_string_parameters(network_ini, 'model_type')
    batch_size = parameters.get_digit_parameters(network_ini,'batch_size', 'int')

    # Initialize the processor
    processor = SamProcessor.from_pretrained(get_model_type_hf(model_type))

    #TODO: how to use valid_images?, or use 100% of image patches for validation
    train_images, valid_images = prepare_dataset_for_SAM_RS(para_file)

    # Create an instance of the SAMDataset for training data (after get sub-images and splitting)
    train_dataset = SAM_RS_Dataset(dataset=train_images, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    num_epochs = parameters.get_digit_parameters(network_ini,'train_epoch_num','int')
    # lr = parameters.get_digit_parameters(network_ini,'base_learning_rate','float')  # 1e-5
    lr = 1e-5
    # weight_decay = parameters.get_digit_parameters(network_ini,'weight_decay','float')
    weight_decay = 0

    # Load the model
    model = SamModel.from_pretrained(get_model_type_hf(model_type))

    # Initialize the optimizer and the loss function
    optimizer = Adam(model.mask_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    # Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

    #  save the model
    save_path = os.path.join(train_save_dir, 'finetuned_%s.pth' % expr_name)
    torch.save(model.state_dict(), save_path)




def fine_tune_sam_main(para_file, pre_train_model='',gpu_num=1):
    print(datetime.now(), "fine-tune Segment Anything models")
    SECONDS = time.time()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    WORK_DIR = os.getcwd()
    trained_model = fine_tune_sam(WORK_DIR, para_file, pre_train_model=pre_train_model, gpu_num=gpu_num)

    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost for training SAM: %.2f seconds">>time_cost.txt' % duration)

def main(options, args):

    para_file = args[0]
    pre_train_model = options.pretrain_model

    fine_tune_sam_main(para_file, pre_train_model=pre_train_model)

if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-04-17")
    parser.description = 'Introduction: fine-tuning segment anything models (huggingface)'

    parser.add_option("-m", "--pretrain_model",
                      action="store", dest="pretrain_model",default='',
                      help="the pre-trained model")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
