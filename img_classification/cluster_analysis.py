#!/usr/bin/env python
# Filename: cluster_analysis.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 July, 2025
"""

import os,sys
from optparse import OptionParser

import torch
import torch.nn.functional as F
import skimage.io as io
import PIL.Image
import numpy as np

from model_visualization import load_clip_model


code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
# import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function


def get_image_path_list(input, file_extension='.tif'):
    image_path_list = []
    if input.endswith(".txt"):
        # read file name from the txt file
        image_path_list = io_function.read_list_from_txt(input)
    elif os.path.isfile(input):
        image_path_list.append(image_path_list)
    elif os.path.isdir(input):
        # read files name from a folder
        image_path_list = io_function.get_file_list_by_pattern(image_path_list, f'*{file_extension}')
    else:
        raise IOError(f'Cannot recognize the input: {image_path_list}')

    return image_path_list

def get_img_features(image_path_list, model, preprocess, device, b_numpy=False):
    """obtain the feature (vector) of images in latent space"""

    feature_list = []

    with torch.no_grad():
        for idx, image_path in enumerate(image_path_list):
            print(f' {idx + 1}/{len(image_path_list)}, generating feature for {os.path.basename(image_path)}')
            image = io.imread(image_path)
            pil_image = PIL.Image.fromarray(image)
            image = preprocess(pil_image).unsqueeze(0).to(device)

            image_feature = model.encode_image(image).to(device, dtype=torch.float32)
            # print(image_feature.shape)
            if b_numpy:
                image_feature_np = image_feature.cpu().numpy().squeeze()
                print(f'feature shape: {image_feature_np.shape}, dtype: {image_feature_np.dtype}')
                print(f'feature min: {image_feature_np.min()}, max: {image_feature_np.max()}, mean: {image_feature_np.mean()}')
            feature_list.append(image_feature_np)

    if b_numpy:
        return np.stack(feature_list)
    else:
        return torch.stack(feature_list)




def dimension_reduction(feature_list, method='PCA', n_components=2):
    """reduce the dimension of feature vectors"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'TSNE':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError(f'Unsupported method: {method}')

    reduced_features = reducer.fit_transform(feature_list)
    return reduced_features

def plot_feature(reduced_features, output_file='reduced_features.png'):
    """plot the reduced features using matplotlib"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
    plt.title('Reduced Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid()
    plt.savefig(output_file)
    plt.close()


def calculate_similarity_matrix(ref_image_features, search_image_features, b_normalize=False, b_scale100=False, apply_softmax=False):
    """
    Calculate the similarity matrix between reference images and search images.

    Args:
        ref_image_features (torch.Tensor): A tensor of shape (n, feature_dim) representing features of n reference images.
        search_image_features (torch.Tensor): A tensor of shape (m, feature_dim) representing features of m search images.
        apply_softmax (bool): Whether to apply the softmax function to the similarity scores.

    Returns:
        torch.Tensor: A similarity matrix of shape (n, m), where each element (i, j) represents the similarity
                      between the i-th reference image and the j-th search image.
    """
    # Ensure the inputs are normalized (common for cosine similarity)
    if b_normalize:
        ref_image_features = F.normalize(ref_image_features, p=2, dim=1)  # Normalize along rows (L2 norm)
        search_image_features = F.normalize(search_image_features, p=2, dim=1)

    # Calculate the similarity matrix using matrix multiplication
    similarity_matrix = ref_image_features @ search_image_features.T  # Shape: (n, m)

    # Scale the similarity (optional, often used in models like CLIP)
    if b_scale100:
        similarity_matrix *= 100.0

    # Apply softmax along the columns (optional)
    if apply_softmax:
        similarity_matrix = F.softmax(similarity_matrix, dim=-1)  # Normalize along the last dimension (columns)

    return similarity_matrix


def test_calculate_similarity_matrix():
    ref_image_list = get_image_path_list('image_list_10.txt')
    search_image_list = get_image_path_list('image_list_10.txt')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = load_clip_model(device, model_type='ViT-B/32', trained_model=None)

    ref_img_features = get_img_features(ref_image_list,clip_model,preprocess,device)
    search_img_features = get_img_features(search_image_list,clip_model,preprocess,device)

    similar_matrix = calculate_similarity_matrix(ref_img_features,search_img_features)
    print(similar_matrix)


def main(options, args):

    test_calculate_similarity_matrix()
    sys.exit(1)

    image_list = get_image_path_list(args[0])
    trained_model = options.trained_model
    model_type = options.model_type

    device = "cuda"  if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = load_clip_model(device, model_type=model_type, trained_model=trained_model)

    image_features = get_img_features(image_list, clip_model, preprocess, device)

    reduce_feature = dimension_reduction(image_features, method='PCA', n_components=2)
    print(f'Reduced feature shape: {reduce_feature.shape}')
    print(f'Reduced feature min: {reduce_feature.min()}, max: {reduce_feature.max()}, mean: {reduce_feature.mean()}')
    # Here you can add code to visualize the reduced features, e.g., using matplotlib or seaborn
    # print(reduce_feature)
    plot_feature(reduce_feature, output_file='reduced_features.png')
    

    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] image_path or image_folder or image_list.txt "
    parser = OptionParser(usage=usage, version="1.0 2025-07-3")
    parser.description = 'Introduction: cluster analysis of small images using AI foundation models '

    parser.add_option("-s", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model")

    parser.add_option("-t", "--model_type",
                      action="store", dest="model_type", default='ViT-B/32',
                      help="the model type or architecture")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)