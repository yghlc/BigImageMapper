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
import gc

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
import basic_src.timeTools as timeTools

from class_utils import create_training_data_from_txt,prepare_training_data, get_data_transforms, load_cnn_models
from prediction_class import run_prediction_cnn, calculate_top_k_accuracy

import logging
logger = logging.getLogger("Model")

def log_string(str):
    logger.info(str)
    print(str)


def run_training(work_dir, network_ini, dataloaders,dataset_sizes,device, model, criterion, optimizer, scheduler, num_epochs=25):

    # setting logger
    logger.setLevel(logging.INFO)
    # Remove and close existing FileHandlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            print(f'Removing existing log file handler: {handler.baseFilename}')
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler('%s/%s.txt' % (work_dir, 'train_log-%s-%s' % ('',timeTools.get_now_time_str())))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    t0 = time.time()
    # Create a temporary directory to save training checkpoints
    best_model_params_path = os.path.join(work_dir, 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        t1 = time.time()
        log_string(f'Training Epoch {epoch+1}/{num_epochs}')
        log_string('-' * 10)

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

            log_string(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        # log_string('\n')

        time_elapsed = time.time() - t1
        log_string(f'Time cost in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        log_string(f'Best val Acc: {best_acc:4f}\n')

        # load best model weights
        state_dict = torch.load(best_model_params_path, weights_only=True)
        model.load_state_dict(state_dict)
        del state_dict

    log_string('\n')
    time_elapsed = time.time() - t0
    log_string(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Clean up GPU memory before returning
    if str(device) != "cpu":
        torch.cuda.empty_cache()

    return model


def train_a_cnn_model(WORK_DIR, para_file, pre_train_model='',train_data_txt='', gpu_num=1):

    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)


    batch_size = parameters.get_digit_parameters(network_ini, 'batch_size', 'int')
    num_workers = parameters.get_digit_parameters(para_file, 'process_num', 'int')

    data_transform = get_data_transforms()


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
    basic.outputlogMessage(f'dataset_sizes: {dataset_sizes}')
    # output the count of each class in the training dataset and validation dataset
    class_counts = {}
    for set_name in ['train', 'val']:
        class_counts[set_name] = {}
        for _, labels, _ in dataloaders[set_name]:
            for label in labels:
                label = label.item()
                if label in class_counts[set_name]:
                    class_counts[set_name][label] += 1
                else:
                    class_counts[set_name][label] = 1 
    basic.outputlogMessage(f'Class counts in training dataset: {class_counts}')

    class_names = image_datasets['train'].classes


    # train our model on an `accelerator, such as CUDA, MPS, MTIA, or XPU.
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model_type = parameters.get_string_parameters(network_ini, 'model_type')
    num_epochs = parameters.get_digit_parameters(network_ini,'train_epoch_num', 'int')

    model_ft = load_cnn_models(model_type)

    b_train_final_layer_only = parameters.get_bool_parameters_None_if_absence(network_ini,'b_train_final_layer_only')
    if b_train_final_layer_only is True:
        for param in model_ft.parameters():
            param.requires_grad = False


    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    learning_rate = parameters.get_digit_parameters(network_ini,'base_learning_rate', 'float') # 0.001
    momentum = parameters.get_digit_parameters(network_ini,'momentum', 'float') # 0.9
    step_size = parameters.get_digit_parameters(network_ini,'step_size', 'int') #7
    gamma = parameters.get_digit_parameters(network_ini,'gamma', 'float') # 0.1

    # print('debuging: learning_rate, momentum, step_size, gamma', learning_rate, momentum, step_size, gamma)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    model_ft = run_training(train_save_dir, network_ini, dataloaders, dataset_sizes,device, 
                            model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


    # run_training(work_dir, network_ini, dataloaders,dataset_sizes,device, model, criterion, optimizer, scheduler, num_epochs=25)


    # evaluate the model on the validation set 
    ###########################################################################
    pre_probs, ground_truths = run_prediction_cnn(model_ft, dataloaders['val'], device)

    top_probs_1, top_labels_1 = pre_probs.cpu().topk(1, dim=-1)
    # top1 accuracy
    top1_acc_save_path = os.path.join(train_save_dir, f'{expr_name}_top1_accuracy.txt' )
    calculate_top_k_accuracy(top_labels_1, ground_truths, save_path=top1_acc_save_path, k=1)

    ###########################################################################



    # copy and back up parameter files
    test_id = os.path.basename(WORK_DIR) + '_' + expr_name
    bak_para_ini = os.path.join(train_save_dir, '_'.join([test_id, 'para']) + '.ini')
    bak_network_ini = os.path.join(train_save_dir, '_'.join([test_id, 'network']) + '.ini')

    io_function.copy_file_to_dst(para_file, bak_para_ini,overwrite=True)
    io_function.copy_file_to_dst(network_ini, bak_network_ini,overwrite=True)

    # Clean up GPU memory after training
    if device != "cpu":
        model_ft = model_ft.cpu()  # Move model to CPU
        del model_ft
        del dataloaders
        del criterion
        del optimizer_ft
        del exp_lr_scheduler
        
        torch.cuda.empty_cache()  # Clear GPU cache
        gc.collect()  # Force garbage collection


def test_different_epoch_numbers(expr_name,WORK_DIR,para_file,pre_train_model,train_data_txt,gpu_num):
    # test with different epoch  numbers
    for epoch in range(10, 500, 20):
    # for epoch in range(10, 100, 20):
        new_exp_name = f'{expr_name}_Epo{epoch}'
        if os.path.isdir(os.path.join(WORK_DIR, new_exp_name)):
            print(f"Directory {new_exp_name} already exists, skipping epoch {epoch}")
            continue

        # update the epoch number in the network ini file
        network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
        parameters.write_Parameters_file(network_ini, 'train_epoch_num', epoch)

        train_a_cnn_model(WORK_DIR, para_file, pre_train_model=pre_train_model,train_data_txt=train_data_txt,gpu_num=gpu_num)

        # move and backup the results
        os.system(f"mv {expr_name} {new_exp_name}")
        os.system(f'rm {os.path.join(WORK_DIR, new_exp_name)}/*.pt')  # remove the checkpoint files to save space

def plot_accuracy_vs_epoch_number(expr_name,WORK_DIR, folder_pattern="Epo"):
    ########################################################################
    # plot the F1 score vs epoch curve
    import matplotlib.pyplot as plt
    accuracy_txt_list = io_function.get_file_list_by_pattern(WORK_DIR, f'{expr_name}_{folder_pattern}*/exp*_top1_accuracy.txt')
    epoch_list = []
    for accuracy_txt in accuracy_txt_list:
        epoch_str = accuracy_txt.split(f'{folder_pattern}')[1].split('/')[0]
        epoch_list.append(int(epoch_str))

    # sorted accuracy_txt_list by epoch
    accuracy_txt_list = [x for _, x in sorted(zip(epoch_list, accuracy_txt_list))]
    for txt in accuracy_txt_list:
        print(txt)

    epoch_list = []  # as accuracy_txt_list has been sorted by epoch, we can re-extract the epoch number to make sure the order is correct
    c1_accuracy_list = []
    c0_accuracy_list = []
    top1_accuracy_list = []
    for accuracy_txt in accuracy_txt_list:
        epoch_str = accuracy_txt.split(f'{folder_pattern}')[1].split('/')[0]
        epoch_list.append(int(epoch_str))
        with open(accuracy_txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('class: 1'):
                    c1_accuracy = float(line.split(':')[3].strip())/100.0
                    c1_accuracy_list.append(c1_accuracy)
                elif line.startswith('class: 0'):
                    c0_accuracy = float(line.split(':')[3].strip())/100.0   
                    c0_accuracy_list.append(c0_accuracy)
                elif line.startswith('top 1 accuracy'):
                    top1_accuracy = float(line.split(':')[2].strip())/100.0
                    top1_accuracy_list.append(top1_accuracy)
                else:
                    pass
                    
    print('mean for c1_accuracy_list', np.mean(c1_accuracy_list))
    print('mean for c0_accuracy_list', np.mean(c0_accuracy_list))
    print('mean for top1_accuracy_list', np.mean(top1_accuracy_list))
    # print(f'Epoch list: {epoch_list}')
    # print(f'F1 scores list: {c1_accuracy_list}')
    # plot the F1 score vs epoch curve
    plt.figure()
    # just plot a scatter plot, since the F1 score may not be monotonic with epoch, and we only have a few points
    # plt.scatter(epoch_list, f1_scores_list, marker='o')
    plt.plot(epoch_list, c1_accuracy_list, marker='+', label='Class 1 Accuracy')
    plt.plot(epoch_list, c0_accuracy_list, marker='x', label='Class 0 Accuracy')
    plt.plot(epoch_list, top1_accuracy_list, marker='o', label='Top 1 Accuracy')

    plt.xlabel(f'{folder_pattern}')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Validation Accuracy vs {folder_pattern}')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(WORK_DIR, f'{expr_name}_validation_accuracy_vs_{folder_pattern}.png'))
    print(f'Finished plotting Validation Accuracy vs {folder_pattern} curve, saved to {expr_name}_validation_accuracy_vs_{folder_pattern}.png')
    ############################################################################################


def test_different_training_sample_count(expr_name,WORK_DIR,para_file,pre_train_model,train_data_txt,gpu_num):
    # test with different sample counts
    for sam_count in range(10, 1000, 10):
        new_exp_name = f'{expr_name}_Sample{sam_count}'
        if os.path.isdir(os.path.join(WORK_DIR, new_exp_name)):
            print(f"Directory {new_exp_name} already exists, skipping sample count {sam_count}")
            continue

        # update the sample count in the network ini file
        parameters.write_Parameters_file(para_file, 'a_few_shot_samp_count', sam_count)

        # generate the training data with the specified sample count
        os.system("rm -r training_data")
        os.system(f"python ~/codes/PycharmProjects/BigImageMapper/img_classification/get_organize_training_data.py {para_file}")

        train_a_cnn_model(WORK_DIR, para_file, pre_train_model=pre_train_model,train_data_txt=train_data_txt,gpu_num=gpu_num)

        # move and backup the results
        os.system(f"mv {expr_name} {new_exp_name}")
        os.system(f'rm {os.path.join(WORK_DIR, new_exp_name)}/*.pt')  # remove the checkpoint files to save space


def test_train_a_cnn_model():
    # set work directory
    if os.path.isfile('run_train.sh'):
        WORK_DIR = os.getcwd()
    else:
        # WORK_DIR= "/home/hlc/Data/slump_demdiff_classify/cnn_rsModel_classify"
        WORK_DIR= os.path.expanduser("~/Data/slump_demdiff_classify/cnn_rsModel_classify_exp14R1") # for testiing differrent sample numbers
        os.chdir(WORK_DIR)

    # para_file = 'main_para_exp14.ini'
    pre_train_model = ""
    train_data_txt = ""
    gpu_num = 1
    
    # train_a_cnn_model(WORK_DIR, para_file, pre_train_model=pre_train_model,train_data_txt=train_data_txt,gpu_num=gpu_num)


    
    ### test_different_epoch_numbers(expr_name,WORK_DIR,para_file,pre_train_model,train_data_txt,gpu_num)
    # expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    # plot_accuracy_vs_epoch_number(expr_name,WORK_DIR)

    ### test on different training samples
    para_file = 'main_para_exp14R1.ini'
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    test_different_training_sample_count(expr_name,WORK_DIR,para_file,pre_train_model,train_data_txt,gpu_num)
    plot_accuracy_vs_epoch_number(expr_name,WORK_DIR, folder_pattern="Sample")

    

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

    test_train_a_cnn_model()
    sys.exit(0)

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