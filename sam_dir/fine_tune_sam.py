#!/usr/bin/env python
# Filename: fine_tune_sam.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 06 April, 2024
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser
import torch

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function

from sam_utils import get_totalmask,load_datasets

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch.utils.tensorboard.writer import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F

# for loss calculation
alpha = 0.8
gamma = 2

class ModelSAM(nn.Module):
    """
    Wrapper for the sam model to fine-tune the model on a new dataset

    ...
    Attributes:
    -----------
    freeze_encoder (bool): freeze the encoder weights
    freeze_decoder (bool): freeze the decoder weights
    freeze_prompt_encoder (bool): freeze the prompt encoder weights
    transform (ResizeLongestSide): resize the images to the model input size

    Methods:
    --------
    setup(): load the model and freeze the weights
    forward(images, points): forward pass of the model, returns the masks and iou_predictions
    """

    def __init__(self, freeze_encoder=True, freeze_decoder=False, freeze_prompt_encoder=True):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        # we need this to make the input image size compatible with the model
        self.transform = ResizeLongestSide(1024) #This is 1024, because sam was trained on 1024x1024 images

    def setup(self, model_type, checkpoint):
        self.model = sam_model_registry[model_type](checkpoint)
        # to speed up training time, we normally freeze the encoder and decoder
        if self.freeze_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.freeze_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        self.transfrom = ResizeLongestSide(self.model.image_encoder.img_size)
    def forward(self, images):
        _, _, H, W = images.shape # batch, channel, height, width
        image_embeddings = self.model.image_encoder(images) # shape: (1, 256, 64, 64)
        # get prompt embeddings without acutally any prompts (uninformative)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        # get low resolution masks and iou predictions
        # mulitmask_output=False means that we only get one mask per image,
        # otherwise we would get three masks per image
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings, # sparse_embeddings shape: (1, 0, 256)
            dense_prompt_embeddings=dense_embeddings, # dense_embeddings shape: (1, 256, 256)
            multimask_output=False,
        )
        # postprocess the masks to get the final masks and resize them to the original image size
        masks = F.interpolate(
            low_res_masks, # shape: (1, 1, 256, 256)
            (H, W),
            mode="bilinear",
            align_corners=False,
        )
        # shape masks after interpolate: torch.Size([1, 1, 1024, 1024])
        return masks, iou_predictions


class FocalLoss(nn.Module):
    """ Computes the Focal loss. """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        inputs = inputs.flatten(0, 2)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):
    """ Computes the Dice loss. """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(0, 2)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / \
               (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


def criterion(x, y, device):
    """ Combined dice and focal loss.
    ARGS:
        x: (torch.Tensor) the model output
        y: (torch.Tensor) the target
    RETURNS:
        (torch.Tensor) the combined loss

    """
    focal, dice = FocalLoss(), DiceLoss()
    y = y.to(device)
    x = x.to(device)
    return 20 * focal(x, y) + dice(x, y)


def train_one_epoch(model, trainloader, optimizer, epoch_idx, tb_writer, device):
    """ Runs forward and backward pass for one epoch and returns the average
    batch loss for the epoch.
    ARGS:
        model: (nn.Module) the model to train
        trainloader: (torch.utils.data.DataLoader) the dataloader for training
        optimizer: (torch.optim.Optimizer) the optimizer to use for training
        epoch_idx: (int) the index of the current epoch
        tb_writer: (torch.utils.tensorboard.writer.SummaryWriter) the tensorboard writer
    RETURNS:
        last_loss: (float) the average batch loss for the epoch

    """
    running_loss = 0.
    for i, (image, path, masks) in enumerate(trainloader):
        image = image.to(device)
        optimizer.zero_grad()
        pred, _ = model(image)
        masks = masks[0].to(device)
        total_mask = get_totalmask(masks)
        pred = pred.to(device)
        loss = criterion(pred, total_mask)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    i = len(trainloader)
    last_loss = running_loss / i
    print(f'batch_loss for batch {i}: {last_loss}')
    tb_x = epoch_idx * len(trainloader) + i + 1
    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    running_loss = 0.
    return last_loss



def fine_tune_sam(WORK_DIR, para_file, pre_train_model='', gpu_num=1,b_evaluate=True):
    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = parameters.get_file_path_parameters(network_ini, 'checkpoint')
    model_type = parameters.get_string_parameters(network_ini, 'model_type')


    epochs = parameters.get_digit_parameters(network_ini,'train_epoch_num','int')
    lr = parameters.get_digit_parameters(network_ini,'base_learning_rate','float')
    weight_decay = parameters.get_digit_parameters(network_ini,'weight_decay','float')

    # model setting up
    best_model = None
    model = ModelSAM()
    model.setup(model_type,checkpoint)
    model.to(device)
    img_size = model.model.image_encoder.img_size

    # get training data (after get sub-images and splitting)
    trainloader, validloader = load_datasets(para_file, img_size=img_size)

    # train SAM
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_loss = float('inf')
    timestamp_writer = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(train_save_dir, f"trainer_{timestamp_writer}"))
    for epch in range(epochs):  # type: ignore
        running_vloss = 0.
        model.train(True)
        avg_batchloss = train_one_epoch(model, trainloader, optimizer, epch, writer, device)
        if not b_evaluate:  # type: ignore
            continue
        with torch.no_grad():
            for images, path, masks in validloader:
                model.to(device)
                images = images.to(device)
                masks = masks[0].to(device)
                total_mask = get_totalmask(masks)
                total_mask = total_mask.to(device)
                model.eval()
                preds, iou = model(images)
                preds = preds.to(device)
                vloss = criterion(preds, total_mask, device)
                running_vloss += vloss.item()
        print(f'epoch: {epch}, validloss: {running_vloss}')
        avg_vloss = running_vloss / len(validloader)
        # save model
        print(f'epoch: {epch}, validloss: {running_vloss}')

        if running_vloss < best_valid_loss:
            best_model = model
            best_valid_loss = running_vloss
        print(f'best valid loss: {best_valid_loss}')

    if best_model is not None:
        # save trained models
        save_path = os.path.join(train_save_dir,'finetuned_%s.pth'%expr_name)
        torch.save(best_model.state_dict(), save_path)

    return best_model


def fine_tune_sam_main(para_file, pre_train_model='',gpu_num=1):
    print(datetime.now(), "fine-tune Segment anything models")
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
    parser = OptionParser(usage=usage, version="1.0 2024-04-06")
    parser.description = 'Introduction: fine-tuning segment anything models '

    parser.add_option("-m", "--pretrain_model",
                      action="store", dest="pretrain_model",default='',
                      help="the pre-trained model")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
