#!/usr/bin/env python
# Filename: model_visualization.py 
"""
introduction: visualize the model output using Grad-CAM or t-SNE

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 20 June, 2025
"""

import os,sys

import torch
import clip
from PIL import Image

from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt
import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic

def heatmap_clip_one_figure(model, preprocess, device, img_path, text_prompt, save_fig=None):
# try:
    # Load and preprocess the image
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    # Tokenize the text prompt
    text_inputs = clip.tokenize([text_prompt]).to(device)

    # Compute image and text features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity scores
    similarity = (image_features @ text_features.T).squeeze(0)
    probs = similarity.softmax(dim=0).cpu().numpy()

    # Initialize Grad-CAM for the visual encoder
    cam_extractor = SmoothGradCAMpp(model.visual)

    # Forward pass and get the CAM for the target index (highest similarity score)
    target_index = similarity.argmax().item()
    activation_map = cam_extractor(target_index, image)

    # Convert activation map to heatmap
    heatmap = to_pil_image(activation_map[0].squeeze(0), mode='F')

    # Convert heatmap to numpy
    heatmap_np = np.array(heatmap)

    # Load the original image
    original_image = Image.open(img_path).convert("RGB")
    original_image_np = np.array(original_image)

    # Resize heatmap to the size of the original image
    heatmap_resized = np.uint8(255 * heatmap_np / (heatmap_np.max() + 1e-8))  # Avoid division by zero
    heatmap_resized = Image.fromarray(heatmap_resized).resize(original_image.size, resample=Image.BILINEAR)

    # Overlay the heatmap on the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image_np)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)  # Adjust alpha for transparency
    plt.axis('off')

    # Save the figure if a path is provided
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()

    # except Exception as e:
    #     print(f"An error occurred: {e}")

def test_heatmap_clip_one_figure():
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Generate heatmap
    heatmap_clip_one_figure(
        model=model,
        preprocess=preprocess,
        device=device,
        img_path="Screenshot_slump_canada.png",
        text_prompt="satellite image of a landslide",
        save_fig="heatmap_slump_overlay.png"
    )

def heatmap_clip_classification(WORK_DIR, para_file, device, img_path, trained_model=None, save_fig=None):

    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)

    model, preprocess = clip.load("ViT-B/32", device=device)

    # load trained model
    if os.path.isfile(trained_model):
        checkpoint = torch.load(open(trained_model, 'rb'), map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])

    if device == "cpu":
        model.eval()
    else:
        model.cuda().eval()  # to download the pre-train models.


def main():
    # Load the model and preprocess function
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    test_heatmap_clip_one_figure()

    pass


if __name__ == '__main__':
    main()
