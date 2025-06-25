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

from torch.utils.data import Dataset, DataLoader

# Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, image_paths, image_classes, preprocess):
        """
        Initialize the dataset.

        Args:
            image_paths (list): List of image file paths.
            image_classes (list): List of class labels corresponding to the images.
            preprocess (callable): Preprocessing function for images (e.g., CLIP preprocess).
        """
        self.image_paths = image_paths
        self.image_classes = image_classes
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the preprocessed image and its class label.
        """
        image_path = self.image_paths[idx]
        label = self.image_classes[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)  # Apply preprocessing
        return image, label

def load_clip_model(device,model_type='ViT-B/32',trained_model=None):
    model, preprocess = clip.load(model_type, device=device)
    # load trained model
    if trained_model is not None:
        if os.path.isfile(trained_model):
            checkpoint = torch.load(open(trained_model, 'rb'), map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'])

    return model, preprocess


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


def heatmap_clip_one_figure(model, preprocess, device, img_path, text_prompts, save_fig=None):
    # Set the model to full precision (float32)
    model.float()

    # Load and preprocess the image
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    # Tokenize the text prompt
    text_inputs = clip.tokenize(text_prompts).to(device)

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

    # # Normalize the heatmap using the 95th percentile
    # percentile_95 = np.percentile(activation_map_np, 95)
    # activation_map_np = np.clip(activation_map_np, 0, percentile_95)  # Clip values at 95th percentile
    # activation_map_np /= activation_map_np.max() + 1e-8  # Normalize to [0, 1]

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

    # text_prompts = ["Photo of a tiger cat","Photo of a dog", "Photo of a car", "Photo of a tree",
    #                 "Photo of a person", "Photo of a building", "Photo of a mountain", "Photo of a river"]

    text_prompts = ["a tiger cat","a dog", "a car", "a tree",
                    "a person", "a building", "a mountain", "a river"]

    # input_img_path = "Screenshot_slump_canada.png"
    # save_fig = "heatmap_slump_overlay.png"
    # text_prompt = "satellite image of a landslide"

    # input_img_path = "a_cat.png"
    # save_fig = "heatmap_cat_overlay.png"
    # text_prompt = "Photo of a cat"

    # 

    input_img_path = "n02123159_tiger_cat.JPEG"
    save_fig = "heatmap_tiger_cat_overlay.png"
    

    # Generate heatmap
    heatmap_clip_one_figure(
        model=model,
        preprocess=preprocess,
        device=device,
        img_path=input_img_path,
        text_prompts=text_prompts,
        save_fig=save_fig
    )


def tSNE_visualiztion(in_features, class_labels, perplexity=30, n_components=2, learning_rate='auto', init='pca',
                      save_fig=None):
    import numpy as np
    from sklearn.manifold import TSNE

    X_embedded = TSNE(n_components=n_components, learning_rate=learning_rate,
                    init=init, perplexity=perplexity).fit_transform(in_features)
    print(X_embedded.shape)

    # plot it X_embedded
    print(f"t-SNE output shape: {X_embedded.shape}")

    # Get unique classes and assign a color to each class
    unique_classes = np.unique(class_labels)
    num_classes = len(unique_classes)
    # Assign the most distinct colors for small numbers of classes
    if num_classes <= 3:
        # Predefined distinct colors for 2 or 3 classes
        predefined_colors = ['red', 'blue', 'green']
        colors = {class_label: predefined_colors[i] for i, class_label in enumerate(unique_classes)}
    else:
        # Use a colormap for larger numbers of classes
        colormap = plt.cm.get_cmap('tab10', num_classes)  # Tab10 colormap
        colors = {class_label: colormap(i) for i, class_label in enumerate(unique_classes)}

    # colors = plt.cm.get_cmap('tab10', num_classes)  # Use a colormap for up to 10 classes (extend if needed)

    # Create a scatter plot with color coding by class
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(unique_classes):
        # Get indices of the samples belonging to the current class
        class_indices = [idx for idx, label in enumerate(class_labels) if label == class_label]
        plt.scatter(
            X_embedded[class_indices, 0],  # t-SNE dim 1 for the current class
            X_embedded[class_indices, 1],  # t-SNE dim 2 for the current class
            c=[colors[i]],  # Assign color for the current class
            label=class_label,  # Use the class label for legend
            alpha=0.6,
            edgecolors='k'
        )

    # Add legend, titles, and labels
    plt.title("t-SNE Visualization of Image Features by Class", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Classes", fontsize=10, loc="best")
    plt.grid(alpha=0.5)

    # Save the plot
    if save_fig is None:
        save_fig = "tsne_visualization.png"
    plt.savefig(save_fig, dpi=200, bbox_inches='tight')
    print(f"t-SNE plot saved to {save_fig}")

    # Clear the figure to save memory if this function is called multiple times
    plt.close()

def cal_clip_text_features(model,label_list_txt,text_des_template="This is a satellite image of a {}"):

    class_labels = [item.split(',')[0] for item in io_function.read_list_from_txt(label_list_txt)]
    text_descriptions = [text_des_template.format(label) for label in class_labels]
    # print(text_descriptions)
    text_tokens = clip.tokenize(text_descriptions).cuda()

    # process text
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_features_np = text_features.cpu().numpy()
    print('text_features_np.shape:', text_features_np.shape)  # text_features_np

    return text_features_np


def cal_clip_image_features(model,preprocess, image_class_txt, image_folder=None,batch_size=256,num_workers=8):
    """
        Calculate CLIP image features using a DataLoader for batch-wise processing.

        Args:
            model: The CLIP model.
            preprocess: The preprocessing pipeline for the CLIP model.
            image_class_txt (str): Path to the text file containing image paths and class labels.
            image_folder (str): Optional folder containing the images.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.

        Returns:
            tuple: Numpy array of image features and a list of class labels.
        """
    # Read image paths and class labels from the text file
    image_list = [item.split() for item in io_function.read_list_from_txt(image_class_txt)]

    # Resolve full paths to images if a folder is specified
    if image_folder is not None:
        image_path_list = [os.path.join(image_folder, item[0]) for item in image_list]
    else:
        image_path_list = [item[0] for item in image_list]

    # Extract class labels
    image_class_list = [int(item[1]) for item in image_list]
    print("image_class_list size:", len(image_class_list))

    # Initialize the custom dataset
    dataset = ImageDataset(image_path_list, image_class_list, preprocess)

    # Initialize the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for feature extraction
        num_workers=num_workers,
        pin_memory=True
    )

    # Process images in batches and calculate features
    image_features_list = []
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (images, _) in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            # Move images to GPU (or appropriate device)
            image_input = images.cuda(non_blocking=True)
            # Calculate image features
            image_features = model.encode_image(image_input).float()
            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # Append features to the list
            image_features_list.append(image_features.cpu().numpy())

    # Concatenate all features into a single numpy array
    image_features_np = np.concatenate(image_features_list, axis=0)
    print('image_features_np.shape:', image_features_np.shape)

    return image_features_np, image_class_list


def tSNE_CLIP_UCM_dataset(device):
    data_dir = os.path.expanduser('~/Data/image_classification/UCMerced_LandUse')

    model, preprocess = load_clip_model(device)

    # read classes info
    label_list_txt = os.path.join(data_dir, 'label_list.txt')
    text_features_np = cal_clip_text_features(model, label_list_txt)

    # read images
    image_txt = os.path.join(data_dir, 'all.txt')
    image_folder = os.path.join(data_dir, 'Images')
    image_features_np, image_class_list = cal_clip_image_features(model,preprocess,image_txt,image_folder=image_folder)

    # tSNE_visualiztion
    for perplexity in range(5, 51, 5):
        print(f"Running t-SNE with perplexity={perplexity}")
        save_fig = f'tsne_UCM_clip_vis_perpl_{perplexity}.png'
        tSNE_visualiztion(
            in_features=image_features_np,
            class_labels=image_class_list,
            perplexity=perplexity,
            save_fig=save_fig
        )


def tSNE_CLIP_S2_slump_images(device):
    data_dir = os.path.expanduser('~/Data/slump_demdiff_classify/clip_classify/training_data_exp12')
    model_dir = os.path.expanduser('~/Data/slump_demdiff_classify/clip_classify/exp12')

    # model, preprocess = load_clip_model(device)

    trained_model = os.path.join(model_dir, 'model_RN50x4_exp12.ckpt')
    model_type = 'RN50x4'  # or 'ViT-B/32', 'RN50', etc.
    model, preprocess = load_clip_model(device, model_type=model_type, trained_model=trained_model)

    # read classes info
    label_list_txt = os.path.expanduser('~/Data/slump_demdiff_classify/label_list_merge_v4.txt')
    text_features_np = cal_clip_text_features(model, label_list_txt)

    # read images
    image_txt = os.path.join(data_dir, 'merge_training_data_for_exp12_from_10_regions_all.txt')
    image_features_np, image_class_list = cal_clip_image_features(model, preprocess, image_txt,
                                                                  image_folder=None)

    if trained_model is not None:
        trained_model_name = io_function.get_name_no_ext(trained_model)
    else:
        trained_model_name = model_type

    # tSNE_visualiztion
    for perplexity in range(5, 201, 5):
        print(f"Running t-SNE with perplexity={perplexity}, {trained_model}, model_type={model_type}")
        save_fig = f'tsne_S2_clip_{trained_model_name}_perpl_{perplexity}.png'
        tSNE_visualiztion(
            in_features=image_features_np,
            class_labels=image_class_list,
            perplexity=perplexity,
            save_fig=save_fig
        )



def test_tSNE_CLIP_visualization(device):

    # t-SNE visualization of features extracted by the visual encoder network from the UCM
    tSNE_CLIP_UCM_dataset(device)

##########################################################################################
# adopt from: https://colab.research.google.com/github/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb

from scipy.ndimage import filters
from torch import nn
import torch.nn.functional as F

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1 * (1 - attn_map ** 0.7).reshape(attn_map.shape + (1,)) * img + \
               (attn_map ** 0.7).reshape(attn_map.shape + (1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, blur=True, save_path='heatmap.jpg'):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.savefig(save_path,dpi=200)
    plt.show()


def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor,
        layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam

def CLIP_GradCAM_Visualization(image_path, image_caption, clip_model='RN50', saliency_layer='layer4',save_fig='heatmap.jpg'):
    image_caption = image_caption  # @param {type:"string"}
    # @markdown ---
    # @markdown #### CLIP model settings
    clip_model = clip_model  # @param ["RN50", "RN101", "RN50x4", "RN50x16"]
    saliency_layer = saliency_layer  # @param ["layer4", "layer3", "layer2", "layer1"]
    # @markdown ---
    # @markdown #### Visualization settings
    blur = True  # @param {type:"boolean"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, jit=False)

    # Download the image from the web.
    # image_path = 'n01622779_great_grey_owl.JPEG'
    # urllib.request.urlretrieve(image_url, image_path)

    image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_np = load_image(image_path, model.visual.input_resolution)
    text_input = clip.tokenize([image_caption]).to(device)

    attn_map = gradCAM(
        model.visual,
        image_input,
        model.encode_text(text_input).float(),
        getattr(model.visual, saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()

    viz_attn(image_np, attn_map, blur, save_path=save_fig)

##########################################################################################


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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # test_heatmap_clip_one_figure()
    # image_path = 'n01622779_great_grey_owl.jpeg'
    # image_caption = "the owl"

    # image_path = 'n02123159_tiger_cat.JPEG'
    # image_caption = "the tiger cat"

    # image_path = 'a_cat.png'
    # image_caption = "a cat"

    # image_path = 'Screenshot_slump_canada.png'
    # image_caption = "satellite image of a landslide"

    image_path = 'intersection88.tif'
    image_caption = 'intersection'
    image_caption = 'the satellite image of intersection'

    save_fig = f'heatmap_{image_caption}.jpg'
    CLIP_GradCAM_Visualization(image_path, image_caption, clip_model='RN50', saliency_layer='layer4',save_fig=save_fig)

    
    # tSNE_visualiztion()
    # test_tSNE_CLIP_visualization(device)
    # tSNE_CLIP_S2_slump_images(device)

    pass


if __name__ == '__main__':
    main()
