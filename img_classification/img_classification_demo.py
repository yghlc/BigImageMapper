#!/usr/bin/env python
# Filename: img_classification_demo.py 
"""
introduction: an example of image classification based on

TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

modified from the above tutorial

# need install pytorch

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 23 May, 2021
"""

# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy

# plt.ion() #  interactive mode for jupiter notebook

img_mean = [0.485, 0.456, 0.406]
img_std= [0.229, 0.224, 0.225]

def imshow(inp, title=None,save_path='example_images.jpg'):
    '''Imshow for Tensor'''
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array(img_mean)
    std = np.array(img_std)

    inp = std*inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

    # plt.show()  # show not interactive mode
    plt.savefig(save_path, dpi=200)
    print('Save example images to %s'%save_path)

def show_some_images(dataloaders,class_names):
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

def train_model(model, dataloaders, device, dataset_sizes,criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)

        # Each epoch has a training and vilidation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train() # set model to training mode
            else:
                model.eval()    # set model to evaluate model

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients, why?
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss/ dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, dataloaders,device,class_names, num_images=6):
    # visualize some model prediction results
    was_training = model.training
    model.eval()
    images_for_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i ,(inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs,1)

            for j in range(inputs.size()[0]):
                images_for_far += 1
                ax = plt.subplot(num_images//2, 2, images_for_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j],save_path='predictions.jpg') #_%d %images_for_far

                if images_for_far == num_images:
                    model.train(mode=was_training)
                    return
            model.train(mode=was_training)



def main():

    ### loading data
    # use torchvision and torch.utils.data packages for loading the data
    # The problem we’re going to solve today is to train a model to classify ants and bees.
    # We have about 120 training images each for ants and bees. There are 75 validation images for each class.
    # Usually, this is a very small dataset to generalize upon, if trained from scratch. Since we are using transfer learning,
    # we should be able to generalize reasonably well.
    #
    # This dataset is a very small subset of imagenet.

    # Data augmentation and normalization for training
    # Just normalization for validation

    data_transforms = {
        'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(img_mean, img_std)]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)])
    }

    data_dir = os.path.expanduser('~/Data/image_classification/hymenoptera_data')
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train','val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers = 4)
                   for x in ['train','val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
    class_names = image_datasets['train'].classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # test, show some images
    # show_some_images(dataloaders,class_names)

    # ##################################################################################
    # # train the model
    # # Finetuning the convnet
    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features

    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs,2)

    # model_ft = model_ft.to(device)
    # criterion = nn.CrossEntropyLoss()

    # # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # model_ft = train_model(model_ft,dataloaders,device,dataset_sizes,criterion,optimizer_ft,exp_lr_scheduler,
    #                        num_epochs=25)
    # visualize_model(model_ft,dataloaders,device,class_names)
    ##################################################################################

    # ConvNet as fixed feature extractor
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv,dataloaders,device,dataset_sizes,criterion,optimizer_conv,
    exp_lr_scheduler,num_epochs=25)

    visualize_model(model_conv,dataloaders,device,class_names)
    pass






if __name__ == '__main__':
    main()
    pass

