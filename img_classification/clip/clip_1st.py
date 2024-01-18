#!/usr/bin/env python


# first example: https://github.com/openai/CLIP
import os
import torch
import clip
from PIL import Image

device = "cuda"  if torch.cuda.is_available() else "cpu"

img_path=os.path.join(os.path.expanduser("~/codes/github_public_repositories/CLIP"), 'CLIP.png')

model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    # what is these two lines for? comment them out, the result is the same
    #image_features = model.encode_image(image)
    #text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image,text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Label probs: [[0.9927   0.004185 0.003016]]
print("Label probs:", probs)

