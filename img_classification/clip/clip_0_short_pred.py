#!/usr/bin/env python

# example of CLIP zero-short prediction : https://github.com/openai/CLIP
import os
import clip
import torch 
from torchvision.datasets import CIFAR100

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# download dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/Data/image_classification"), download=True)

# Prepare the inputs 
image, class_id = cifar100[3618]
image_input=preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    #print(image_features,text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
#print(image_features,text_inputs)

# @  matrix multiplication for tensor
# * element-wise multiplication
# Applies the Softmax function to an n-dimensional input Tensor rescaling them 
# so that the elements of the n-dimensional output Tensor lie in the range [0,1] 
# and sum to 1.

similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#print(similarity)
# topk: Returns the k largest elements of the given input tensor along a given dimension.
values, indices = similarity[0].topk(5)

# print the result
print("\n Top predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

print("original:",cifar100.classes[class_id])
