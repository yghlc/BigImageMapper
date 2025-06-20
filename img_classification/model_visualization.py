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

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic


# Custom Grad-CAM class to handle tuple outputs
class CustomSmoothGradCAMpp(SmoothGradCAMpp):
    def _hook_a(self, module, input, output, idx=None):
        """
        Override the activation hook to handle tuple outputs.
        """
        # If the output is a tuple, take the first element; otherwise, use the output as-is
        if isinstance(output, tuple):
            output = output[0]  # Extract the tensor we need
        self.hook_a[idx] = output.data if idx is not None else output.data

    def _hook_g(self, module, input, output, idx=None):
        """
        Override the gradient hook to handle tuple outputs.
        """
        # If the output is a tuple, take the first element; otherwise, use the output as-is
        if isinstance(output, tuple):
            output = output[0]  # Extract the tensor we need
        # Register the gradient hook on the tensor
        self.hook_handles.append(output.register_hook(partial(self._store_grad, idx=idx)))


def heatmap_clip_one_figure(model, preprocess, device, img_path, text_prompt, save_fig=None):
    # Set the model to full precision (float32)
    model.float()

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

    print(f"Similarity scores: {similarity.cpu().numpy()}")
    print(f"Probabilities: {probs}")
    # print(f"model visual:",model.visual)

    # Extract the last attention layer of the Vision Transformer
    target_layer = model.visual.transformer.resblocks[-1].attn

    # Initialize the custom Grad-CAM with the extracted target layer
    cam_extractor = CustomSmoothGradCAMpp(model.visual, target_layer=target_layer)

    # Forward pass to get the CAM for the target index (highest similarity score)
    target_index = similarity.argmax().item()
    out = model.encode_image(image)

    # Generate the activation map
    activation_map = cam_extractor(target_index, out)

      # Convert activation map to numpy for normalization
    activation_map_np = activation_map[0].squeeze(0).cpu().numpy()

    # Normalize the heatmap using the 95th percentile
    percentile_95 = np.percentile(activation_map_np, 95)
    activation_map_np = np.clip(activation_map_np, 0, percentile_95)  # Clip values at 95th percentile
    activation_map_np /= activation_map_np.max() + 1e-8  # Normalize to [0, 1]

    # Convert normalized heatmap to a PIL image
    heatmap = to_pil_image(activation_map_np, mode='F')

    # Convert activation map to heatmap
    # heatmap = to_pil_image(activation_map[0].squeeze(0), mode='F')

    # Convert heatmap to numpy
    heatmap_np = np.array(heatmap)
    print(f"Heatmap shape: {heatmap_np.shape}, maximum value: {heatmap_np.max()}, minimum value: {heatmap_np.min()}, mean value: {heatmap_np.mean()}")

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
        

def test_heatmap_clip_one_figure():
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device:', device)
    model, preprocess = clip.load("ViT-B/32", device=device)

    print('model dtype:', next(model.parameters()).dtype)

    # Convert the model to full precision (float32)
    # model.float()

    # input_img_path = "Screenshot_slump_canada.png"
    # save_fig = "heatmap_slump_overlay.png"
    # text_prompt = "satellite image of a landslide"

    # input_img_path = "a_cat.png"
    # save_fig = "heatmap_cat_overlay.png"
    # text_prompt = "Photo of a cat"

    # 

    input_img_path = "n02123159_tiger_cat.JPEG"
    save_fig = "heatmap_tiger_cat_overlay.png"
    text_prompt = "Photo of a tiger cat"

    # Generate heatmap
    heatmap_clip_one_figure(
        model=model,
        preprocess=preprocess,
        device=device,
        img_path=input_img_path,
        text_prompt=text_prompt,
        save_fig=save_fig
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
