#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Introduction: fine-tune the Geospatial Foundation Models (GFMs) or rsBigModel for a specific task, e.g., 
image classification, object detection, semantic segmentation, etc.

Author: Huang Lingcao
Email: huanglingcao@gmail.com
Created: 2026-03-26
"""
import gc
import os,sys
from optparse import OptionParser

import numpy as np
import torch

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
from img_classification.class_utils import prepare_training_data
import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function
import basic_src.timeTools as timeTools

from rsBigModel_utils import RSPatchTxtDataset, RSPatchTxtModule, get_data_transforms, prepare_train_val_txt

import logging
logger = logging.getLogger("rsBigModel")

def log_string(str):
    logger.info(str)
    print(str)

def fine_tune_rsBigModel_classification(WORK_DIR, para_file, pre_train_model=None, train_data_txt=None):
    # load training data


    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    # print(f'Experiment name: {expr_name}')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)


    batch_size = parameters.get_digit_parameters(network_ini, 'batch_size', 'int')
    num_workers = parameters.get_digit_parameters(para_file, 'process_num', 'int')

    data_transform = get_data_transforms(be_normalzed=True)  # use default RGB normalization for pretrained backbones
    class_labels_txt = parameters.get_file_path_parameters(para_file, 'class_labels')
    # print(class_labels_txt)

    # image_path_list, image_labels, label_txt = 'label.txt', transform=None, split: str = 'train'
    if train_data_txt is not None and len(train_data_txt) > 0:
        train_txt = train_data_txt
        valid_txt = train_data_txt   # or require a separate valid txt
    else:
        train_txt, valid_txt = prepare_train_val_txt(WORK_DIR, para_file)
    # print(f'train_txt: {train_txt}')
    # print(f'valid_txt: {valid_txt}')
    if train_txt is None or valid_txt is None:
        log_string('Error: training txt files are not prepared. Please run img_classification/get_organize_training_data.py first to prepare and organize the training data')
        return False

    fit_ckpt_path = None
    if pre_train_model is not None and len(pre_train_model) > 0:
        if os.path.isfile(pre_train_model) is False:
            raise ValueError('pre_train_model does not exist: %s' % pre_train_model)
        fit_ckpt_path = pre_train_model
        log_string('Resume training from checkpoint: %s' % pre_train_model)

    # ---------------------------------------------------------------------------
    # Building the TerraTorch training pipeline
    # ---------------------------------------------------------------------------
    # Next, the TerraTorch training pipeline for this classification task is built.
    # We initialize a new datamodule using the full UCMerced dataset.
    basic.outputlogMessage('Warning, the test and validation txt files are the same, so the accuracy is the validation accuracy.')
    data_module = RSPatchTxtModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train_txt=train_txt,
        valid_txt=valid_txt,
        test_txt=valid_txt,  # use the validation set for testing as well since we don't have a separate test set
        label_txt=class_labels_txt,
        transforms=data_transform
    )

    data_module.setup("fit")

    train_dataset = data_module.train_dataset
    num_classes = len(train_dataset.classes)
    print(f"Available samples in the training dataset: {len(train_dataset)}, number of classes: {num_classes}")
    
    # return False

    # Plot a few sample images to check the data loading and transformations
    # import matplotlib.pyplot as plt
    # for i in range(0, 300, 100):
    #     print('sample:', i)
    #     print(train_dataset[i]['filename'], train_dataset[i]['label'])
    #     print(train_dataset[i]['image'].shape, 'dtype: {}'.format(train_dataset[i]['image'].dtype),
    #           'min: {}, max: {}'.format(train_dataset[i]['image'].min(), train_dataset[i]['image'].max()),
    #           'mean: {}, std: {}'.format(train_dataset[i]['image'].mean(), train_dataset[i]['image'].std()))
    #     train_dataset.plot(train_dataset[i])
    #     plt.savefig(f"{train_save_dir}/sample_{i}.png")




    # train our model on an `accelerator, such as CUDA, MPS, MTIA, or XPU.
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model_type = parameters.get_string_parameters(network_ini, 'model_type')
    num_epochs = parameters.get_digit_parameters(network_ini,'train_epoch_num', 'int')

    learning_rate = parameters.get_digit_parameters(network_ini,'base_learning_rate', 'float') # 0.001
    # momentum = parameters.get_digit_parameters(network_ini,'momentum', 'float') # 0.9
    # step_size = parameters.get_digit_parameters(network_ini,'step_size', 'int') #7
    # gamma = parameters.get_digit_parameters(network_ini,'gamma', 'float') # 0.1
    b_freeze_backbone = parameters.get_bool_parameters(network_ini,'b_freeze_backbone') # False


    # ---------------------------------------------------------------------------
    # initialize the Lightning Trainer. TerraTorch builds on standard Lightning
    # components, allowing us to use the regular Trainer and callbacks for fine-tuning.
    # The training and validation logic itself is defined by EO-specific task classes
    # provided by TerraTorch, which we initialize in the next step.
    # ---------------------------------------------------------------------------

    import lightning.pytorch as pl
    pl.seed_everything(0)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{train_save_dir}/",
        filename=f"best-val-{model_type}",
        save_weights_only=False, # also save the optimizer state, epoch, etc. for resuming training
        monitor="val/loss",
        mode="min", 
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        precision="bf16-mixed",   # speed up training with mixed precision
        max_epochs=num_epochs,            # train only one epoch for tutorial purposes
        logger=True,              # uses TensorBoard by default
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
        # callbacks=[checkpoint_callback, early_stopping_callback, pl.callbacks.RichProgressBar()],
        default_root_dir=f"{train_save_dir}/",
    )

    from terratorch.tasks import ClassificationTask
    # backbone is prithvi_eo_v2_300
    model = ClassificationTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": model_type.lower(),  # use the prithvi_eo_v2_300 backbone
            "backbone_pretrained": True,    # load pretrained weights
            "backbone_bands": ["BLUE", "GREEN", "RED"],
            "decoder": "IdentityDecoder",   # no decoder is used
            "head_dropout": 0.1,            # dropout in the classification head
            "head_dim_list": [384, 128],    # hidden dimension of the head
            "num_classes": num_classes,              #
        },
        loss="ce",
        optimizer="AdamW",
        ignore_index=-1,
        lr=learning_rate,                            # optimal lr varies; test values between 1e-5 and 1e-4
        freeze_backbone=b_freeze_backbone,
    )

    # Start training)
    trainer.fit(model, datamodule=data_module, ckpt_path=fit_ckpt_path)

    ###########################################################################
    # Prepare the test split
    data_module.setup("test")
    test_dataset = data_module.test_dataset
    print(f"Number of samples in the test dataset: {len(test_dataset)}")
    # Evaluate the model on the test set
    test_results = trainer.test(model=model, datamodule=data_module)
    # save to a file
    test_accuracy_file = os.path.join(train_save_dir, f"{expr_name}_test_accuracy.txt")
    with open(test_accuracy_file, 'w') as f:
        if len(test_results) > 0:
            metrics = test_results[0]
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                f.write(f"{key}: {value}\n")
        else:
            f.write("No test metrics were returned by trainer.test().\n")

        # output the number of samples for each class in the test split
        f.write("\nTest class counts (ground truth):\n")
        if hasattr(test_dataset, 'labels') and hasattr(test_dataset, 'classes'):
            class_counts = np.bincount(test_dataset.labels, minlength=len(test_dataset.classes))
            for class_name, count in zip(test_dataset.classes, class_counts):
                f.write(f"{class_name}: {int(count)}\n")
        else:
            f.write("Class count information is unavailable for this dataset type.\n")

    ###########################################################################


    # copy and back up parameter files
    test_id = os.path.basename(WORK_DIR) + '_' + expr_name
    bak_para_ini = os.path.join(train_save_dir, '_'.join([test_id, 'para']) + '.ini')
    bak_network_ini = os.path.join(train_save_dir, '_'.join([test_id, 'network']) + '.ini')

    io_function.copy_file_to_dst(para_file, bak_para_ini,overwrite=True)
    io_function.copy_file_to_dst(network_ini, bak_network_ini,overwrite=True)

    # Clean up GPU memory after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def test_fine_tune_rsBigModel_classification():
    # change dir
    WORK_DIR= "/home/hlc/Data/slump_demdiff_classify/cnn_rsModel_classify"
    os.chdir(WORK_DIR)
    para_file = 'main_para_exp15.ini'
    pre_train_model = None
    train_data_txt = None
    # fine_tune_rsBigModel_classification(WORK_DIR, para_file, pre_train_model=pre_train_model, train_data_txt=train_data_txt)
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    # test with different epoch  numbers
    for epoch in range(10, 500, 20):
        new_exp_name = f'exp15_Epo{epoch}'
        if os.path.isdir(os.path.join(WORK_DIR, new_exp_name)):
            print(f"Directory {new_exp_name} already exists, skipping epoch {epoch}")
            continue

        # update the epoch number in the network ini file
        network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
        parameters.write_Parameters_file(network_ini, 'train_epoch_num', epoch)
        fine_tune_rsBigModel_classification(WORK_DIR, para_file, pre_train_model=pre_train_model, train_data_txt=train_data_txt)

        # move and backup the results
        
        os.system(f"mv {expr_name} {new_exp_name}")
        os.system(f'rm {os.path.join(WORK_DIR, new_exp_name)}/*.ckpt')  # remove the checkpoint files to save space

def fine_tune_rsBigModel_main(para_file, pre_train_model=None, train_data_txt=None, task_type=None):
    '''
    fine-tune the rsBigModel for a specific task, e.g., image classification, object detection, semantic segmentation, etc.
    :param para_file:
    :param pre_train_model:
    :param train_data_txt:
    :param task_type:
    :return:
    '''

    WORK_DIR = os.getcwd()
    if task_type == 'classification':
        fine_tune_rsBigModel_classification(WORK_DIR, para_file, pre_train_model=pre_train_model, train_data_txt=train_data_txt)
    elif task_type == 'segmentation':
        print('Segmentation fine-tuning is not implemented yet.')
    else:
        print('Error: unsupported task type: {}'.format(task_type))
        return False

def main(options, args):

    para_file = args[0]
    pre_train_model = options.pretrain_model
    train_data_txt = options.train_data_txt
    task_type = options.task_type

    fine_tune_rsBigModel_main(para_file,pre_train_model=pre_train_model,train_data_txt=train_data_txt, task_type=task_type)


if __name__ == "__main__":

    test_fine_tune_rsBigModel_classification()
    sys.exit(0)

    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2026-03-26")
    parser.description = 'Introduction: fine-tune the RS big models using custom data'

    parser.add_option("-m", "--pretrain_model",
                      action="store", dest="pretrain_model",default='',
                      help="the pre-trained model")

    parser.add_option("-t", "--train_data_txt",
                      action="store", dest="train_data_txt",default='',
                      help="the training dataset saved in txt")
    
    parser.add_option("", "--task_type",
                      action="store", dest="task_type",default='',
                      help="the task type, should be one of 'classification', 'segmentation', etc ")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)