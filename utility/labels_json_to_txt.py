#!/usr/bin/env python
# Filename: labels_json_to_txt.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 August, 2025
"""

import os,sys
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function

def get_a_list(in_list, class_idx):
    save_line_list = []
    for v in in_list:
        # text_prompt = v.rstrip('.').replace(',','-')
        text_prompt = v.rstrip('.')
        print(text_prompt)
        save_line_list.append(f'"{text_prompt}", {class_idx}')
        class_idx += 1

    return save_line_list, class_idx

def main():
    json= 'dem_diff_color_text_prompts_from_GPT5.json'
    data_dict = io_function.read_dict_from_txt_json(json)
    save_line_list = []
    print(data_dict['classes'].keys())
    select_keys = ['positive','variants','negative','hard_negatives']

    class_idx = 0

    out_list, class_idx = get_a_list(data_dict['classes']['positive'], class_idx)
    save_line_list.extend(out_list)

    out_list, class_idx = get_a_list(list(data_dict['variants'].values()), class_idx)
    save_line_list.extend(out_list)

    out_list, class_idx = get_a_list(data_dict['classes']['negative'], class_idx)
    save_line_list.extend(out_list)

    out_list, class_idx = get_a_list(data_dict['hard_negatives'], class_idx)
    save_line_list.extend(out_list)

    io_function.save_list_to_txt('label_list_GPT5_text_prompts.txt',save_line_list)



if __name__ == '__main__':
    main()
