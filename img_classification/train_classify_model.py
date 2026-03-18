#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Introduction: to train a image classification model (CNN-based) for the image classification task

Author: Huang Lingcao
Email: huanglingcao@gmail.com
Created: 2026-03-17
"""

import os,sys
from optparse import OptionParser
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function

from class_utils import create_training_data_from_txt,prepare_training_data


def run_training(work_dir, network_ini, dataloaders,dataset_sizes,device, model, criterion, optimizer, scheduler, num_epochs=25):

    t0 = time.time()
    # Create a temporary directory to save training checkpoints
    best_model_params_path = os.path.join(work_dir, 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        # print()

        time_elapsed = time.time() - t0
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model


def train_a_cnn_model(WORK_DIR, para_file, pre_train_model='',train_data_txt='', gpu_num=1):

    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)


    batch_size = parameters.get_digit_parameters(network_ini, 'batch_size', 'int')
    num_workers = parameters.get_digit_parameters(para_file, 'process_num', 'int')

    # simple data resize and normalization for both training and validtion. (no data augmentation) 
    data_transform = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
    # # Data augmentation and normalization for training
    # # Just normalization for validation
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }


    if os.path.isfile(train_data_txt):
        train_dataset = create_training_data_from_txt(para_file, train_data_txt, data_transform, test=False)
    else:
        train_dataset, valid_dataset = prepare_training_data(WORK_DIR, para_file, data_transform, test=False)
        if train_dataset is None:
            return None

    if valid_dataset is None:
        basic.outputlogMessage('Warning, the training data and validation are the same')
        valid_dataset = train_dataset


    image_datasets = {'train': train_dataset, 'val': valid_dataset}
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train','val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes


    # train our model on an `accelerator, such as CUDA, MPS, MTIA, or XPU.
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model_type = parameters.get_string_parameters(network_ini, 'model_type')
    num_epochs = parameters.get_digit_parameters(network_ini,'train_epoch_num', 'int')

    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = run_training(train_save_dir, network_ini, dataloaders, dataset_sizes,device, 
                            model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


    # run_training(work_dir, network_ini, dataloaders,dataset_sizes,device, model, criterion, optimizer, scheduler, num_epochs=25)



    # copy and back up parameter files
    test_id = os.path.basename(WORK_DIR) + '_' + expr_name
    bak_para_ini = os.path.join(train_save_dir, '_'.join([test_id, 'para']) + '.ini')
    bak_network_ini = os.path.join(train_save_dir, '_'.join([test_id, 'network']) + '.ini')

    io_function.copy_file_to_dst(para_file, bak_para_ini,overwrite=True)
    io_function.copy_file_to_dst(network_ini, bak_network_ini,overwrite=True)


def cnn_train_main(para_file, pre_train_model='', train_data_txt='', b_a_few_shot=False, gpu_num=1):
    print(datetime.now(),"train CNN models for image classification")
    SECONDS = time.time()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    WORK_DIR = os.getcwd()
    train_a_cnn_model(WORK_DIR, para_file, pre_train_model=pre_train_model,train_data_txt=train_data_txt,gpu_num=gpu_num)

    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of training: %.2f seconds">>time_cost.txt' % duration)

def main(options, args):

    para_file = args[0]
    pre_train_model = options.pretrain_model
    train_data_txt = options.train_data_txt
    b_a_few_shot = options.b_a_few_shot

    cnn_train_main(para_file,pre_train_model=pre_train_model,train_data_txt=train_data_txt,
                    b_a_few_shot=b_a_few_shot)



if __name__ == "__main__":
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-01-24")
    parser.description = 'Introduction: fine-tune the clip model using custom data'

    parser.add_option("-m", "--pretrain_model",
                      action="store", dest="pretrain_model",default='',
                      help="the pre-trained model")

    parser.add_option("-t", "--train_data_txt",
                      action="store", dest="train_data_txt",default='',
                      help="the training dataset saved in txt")

    parser.add_option("-f", "--b_a_few_shot",
                      action="store_true", dest="b_a_few_shot", default=False,
                      help="if set, will force to run a few shot training, ignoring the the setting in ini files")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)