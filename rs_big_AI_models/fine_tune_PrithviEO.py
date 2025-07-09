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


import matplotlib.pyplot as plt
import h5py

All_BANDS = Landslide4SenseNonGeo.all_band_names

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

def get_deeplearning_model(image_bands, learning_rate=1.0e-4, weight_decay=0.1, b_freeze_backbone=False, head_dropout=0.1):

    backbone_args = dict(
        backbone_pretrained=True,
        backbone="prithvi_eo_v2_300",
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
        devices=1,
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

    return checkpoint_callback, trainer


def load_image_h5(image_path):
    with h5py.File(image_path, "r") as f:
        img = f["img"][:]
    return img

from albumentations import Resize
from albumentations.pytorch.transforms import ToTensorV2


# Preprocess the image
def preprocess_image(img, bands):
    # Extract the required bands
    band_indices = [All_BANDS.index(band) for band in bands]
    img_bands = img[band_indices, :, :]

    # Normalize the image (if required)
    img_bands = img_bands / 255.0  # Assuming the data is in the range [0, 255]

    # Resize the image to match the model input size
    transform = Resize(224, 224)
    transformed = transform(image=img_bands.transpose(1, 2, 0))
    img_resized = transformed["image"]

    # Convert to tensor
    to_tensor = ToTensorV2()
    img_tensor = to_tensor(image=img_resized)["image"]

    return img_tensor.unsqueeze(0)  # Add batch dimension


# Run inference

def predict_image(model, img_tensor):
    print(f"Input tensor dtype: {img_tensor.dtype}")
    img_tensor = img_tensor.to(torch.float32) 
    with torch.no_grad():
        outputs = model(img_tensor)
        print(outputs)
        preds = torch.argmax(outputs.output, dim=1).squeeze(0).cpu().numpy()
        print("preds:",preds)
    return preds


# Visualize the image and prediction
def visualize_results(original_img, prediction_mask):
    plt.figure(figsize=(10, 5))

    # Original image (select RGB channels for visualization)
    plt.subplot(1, 2, 1)
    plt.title("Original Image (RGB)")
    rgb_img = original_img[[All_BANDS.index("RED"), All_BANDS.index("GREEN"), All_BANDS.index("BLUE")], :, :].transpose(1, 2, 0)
    plt.imshow(rgb_img)
    plt.axis("off")

    # Prediction mask
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(prediction_mask, cmap="jet")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("prediction_results.png")
    # plt.show()


def predict_PrithviEO_for_segment(dl_model, image_path, image_bands, ckpt_path=None):
    """
    Predict using the fine-tuned PrithviEO model for segmentation tasks.
    """
    # if ckpt_path is not None:
    #     dl_model = dl_model.load_from_checkpoint(ckpt_path)

    # # Set the model to evaluation mode
    # dl_model.eval()
    original_image = load_image_h5(image_path)
    img_tensor = preprocess_image(original_image, bands=image_bands)
    print(f"Image shape after preprocessing: {img_tensor.shape}")

    # Run prediction
    prediction = predict_image(dl_model, img_tensor)

    # Visualize results
    visualize_results(original_image, prediction)


    pass
    # return predictions

def test_fine_tune_PrithviEO_for_segment():
    DATASET_PATH = os.path.expanduser("~/Data/public_data_AI/Landslide4Sense/data")
    image_bands = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]

    # set up data loader
    batch_size = 16
    num_workers = 8
    data_module = get_data_module(DATASET_PATH,image_bands,batch_size=batch_size, num_workers=num_workers)

    # setting up deep learning model
    learning_rate = 1.0e-4
    weight_decay = 0.1
    FREEZE_BACKBONE = False
    head_dropout = 0.1
    dl_model = get_deeplearning_model(image_bands,learning_rate,weight_decay, b_freeze_backbone=FREEZE_BACKBONE,head_dropout=head_dropout)

    task_name = 'map_landslide'
    OUT_DIR = './'
    EPOCHS = 10
    check_point, trainer = fine_tune_PrithviEO_for_segment(dl_model, data_module, EPOCHS, OUT_DIR, task_name)

    ckpt_path = check_point.best_model_path

    # Test results
    test_results = trainer.test(dl_model, datamodule=data_module, ckpt_path=ckpt_path)

    predict_output = trainer.predict(dl_model, datamodule=data_module, ckpt_path=ckpt_path)
    print("predict_output:", predict_output)

def test_predict_PrithviEO_for_segment():

    image_bands = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
    CKPT_PATH = './map_landslide/checkpoints/best-checkpoint-epoch=09-val_loss=0.00.ckpt'
    image_path = os.path.expanduser('~/Data/public_data_AI/Landslide4Sense/TestData/img/image_374.h5')
    mask_path = os.path.expanduser('~/Data/public_data_AI/Landslide4Sense/TestData/mask/mask_374.h5')


    # ckpt_path = 
    dl_model = SemanticSegmentationTask.load_from_checkpoint(CKPT_PATH, map_location=torch.device('cpu'))
    dl_model.eval()
    # dl_model = dl_model.to(torch.float32)
    # print(dl_model)

    predict_PrithviEO_for_segment(dl_model, image_path, image_bands, ckpt_path=None)

    

    pass


def main():

    test_fine_tune_PrithviEO_for_segment()
    # test_predict_PrithviEO_for_segment()

    pass


if __name__ == '__main__':
    main()
