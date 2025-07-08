#!/usr/bin/env python
# Filename: fine_tune_PrithviEO.py
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 July, 2025
"""

import os,sys
import numpy as np
import torch

import terratorch
from terratorch.datamodules import Landslide4SenseNonGeoDataModule
from terratorch.datasets import Landslide4SenseNonGeo
from terratorch.tasks import SemanticSegmentationTask

import albumentations

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def get_data_module(data_dir, image_bands, batch_size=16,num_workers=8):

    transforms = [
        albumentations.Resize(224, 224),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]

    # Adding augmentations for training
    train_transforms = [
        albumentations.HorizontalFlip(),
        albumentations.Resize(224, 224),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]

    data_module = Landslide4SenseNonGeoDataModule(
        batch_size=batch_size,
        bands=image_bands,
        data_root=data_dir,
        train_transform=train_transforms,
        val_transforms=transforms,
        test_transforms=transforms,
        num_workers=num_workers,
    )

    return data_module

def get_deeplearning_model(image_bands,learning_rate, weight_decay, b_freeze_backbone=False, head_dropout=0.1):

    backbone_args = dict(
        backbone_pretrained=True,
        backbone="prithvi_eo_v2_600_tl",
        # prithvi_eo_v2_300, prithvi_eo_v2_300_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
        backbone_bands=image_bands,
        backbone_num_frames=1,
    )

    decoder_args = dict(
        decoder="UperNetDecoder",
        decoder_channels=256,
        decoder_scale_modules=True,
    )

    necks = [
        dict(
            name="SelectIndices",
            # indices=[2, 5, 8, 11]    # indices for prithvi_eo_v1_100
            indices=[5, 11, 17, 23],  # indices for prithvi_eo_v2_300
            # indices=[7, 15, 23, 31]  # indices for prithvi_eo_v2_600
        ),
        dict(
            name="ReshapeTokensToImage",
        )
    ]


    model_args = dict(
        **backbone_args,
        **decoder_args,
        num_classes=2,
        head_dropout=head_dropout,
        head_channel_list=[128, 64],
        necks=necks,
        rescale=True,
    )

    model = SemanticSegmentationTask(
        model_args=model_args,
        plot_on_val=False,
        loss="focal",
        lr=learning_rate,
        optimizer="AdamW",
        scheduler="StepLR",
        scheduler_hparams={"step_size": 10, "gamma": 0.9},
        optimizer_hparams=dict(weight_decay=weight_decay),
        ignore_index=-1,
        freeze_backbone=b_freeze_backbone,
        freeze_decoder=False,
        model_factory="EncoderDecoderFactory",
    )

    return model


def fine_tune_PrithviEO_for_segment(dl_model, data_module, EPOCHS, OUT_DIR, task_name:str):

    SEED = 0
    pl.seed_everything(SEED)

    logger = TensorBoardLogger(
        save_dir=OUT_DIR,
        name=task_name,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        dirpath=os.path.join(OUT_DIR, task_name, "checkpoints"),
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        devices="auto",
        precision="bf16-mixed",
        num_nodes=1,
        logger=logger,
        max_epochs=EPOCHS,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(dl_model, datamodule=data_module)


def test_fine_tune_PrithviEO_for_segment():
    DATASET_PATH = os.path.expanduser("~/codes/github_public_repositories/Prithvi-EO-2.0/examples/data")
    image_bands = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]

    # set up data loader
    batch_size = 16
    num_workers = 8
    data_module = get_data_module(DATASET_PATH,image_bands,batch_size=batch_size, num_workers=num_workers)

    # setting up deep learning model
    learning_rate = 0.4
    weight_decay = 0.1
    FREEZE_BACKBONE = False
    head_dropout = 0.1
    dl_model = get_deeplearning_model(image_bands,learning_rate,weight_decay, b_freeze_backbone=FREEZE_BACKBONE,head_dropout=head_dropout)

    task_name = 'landslide'
    OUT_DIR = './'
    EPOCHS = 10
    fine_tune_PrithviEO_for_segment(dl_model, data_module, EPOCHS, OUT_DIR, task_name)


def main():

    test_fine_tune_PrithviEO_for_segment()

    pass


if __name__ == '__main__':
    main()
