#!/usr/bin/env python

# example of CLIP Linear-probe evaluation: https://github.com/openai/CLIP
import os
import clip
import torch 

import numpy as np 
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# 
root = os.path.expanduser('~/Data/image_classification')
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)

def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images,labels in tqdm(DataLoader(dataset,batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# calcualte the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)


classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# evaludate 
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float))*100.

print(f"Accuray = {accuracy:.3f}")