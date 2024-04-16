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


def main():
    # Load the state_dicts from the two .pth files
    state_dict1 = torch.load(os.path.expanduser('~/Data/models_deeplearning/segment-anything/sam_vit_h_4b8939.pth') )
    state_dict2 = torch.load('exp5/finetuned_exp5.pth')

    # Compare the key names
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # Check if the key names are equal
    if keys1 == keys2:
        print("The key names in the two .pth files are the same.")
    else:
        print("The key names in the two .pth files are different.")

    with open('keys1.txt','w')  as f_obj:
        f_obj.writelines([str(key) +'\n' for key in keys1])

    with open('keys2.txt','w')  as f_obj:
        f_obj.writelines([str(key) +'\n' for key in keys2])


if __name__ == '__main__':
    main()

