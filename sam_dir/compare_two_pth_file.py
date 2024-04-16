#!/usr/bin/env python
# Filename: compare_two_pth_file.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 16 April, 2024
"""
import os.path

import torch

from fine_tune_sam import ModelSAM

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the state_dicts from the two .pth files
    checkpoint = os.path.expanduser('~/Data/models_deeplearning/segment-anything/sam_vit_h_4b8939.pth')
    state_dict1 = torch.load(checkpoint,map_location=torch.device(device) )

    state_dict2 = torch.load(os.path.expanduser('~/Data/dem_diff_segment/exp5/finetuned_exp5.pth'),map_location=torch.device(device))
    model_trained = ModelSAM()
    model_trained.setup('vit_h', checkpoint)
    model_trained.load_state_dict(state_dict2)
    # state_dict_trained = model_trained.state_dict()

    # Compare the key names
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    keys3 = set(model_trained.model.state_dict().keys())

    # Check if the key names are equal
    if keys1 == keys2:
        print("The key names in the two .pth files are the same.")
    else:
        print("The key names in the two .pth files are different.")

    if keys1 == keys3:
        print("The key1 and key3 are the same.")
    else:
        print("The key1 and key3 are the different.")

    with open('keys1.txt','w')  as f_obj:
        f_obj.writelines([str(key) +'\n' for key in keys1])

    with open('keys2.txt','w')  as f_obj:
        f_obj.writelines([str(key) +'\n' for key in keys2])

    with open('keys3.txt','w')  as f_obj:
        f_obj.writelines([str(key) +'\n' for key in keys3])


if __name__ == '__main__':
    main()

