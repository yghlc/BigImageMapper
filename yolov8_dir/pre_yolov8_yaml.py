#!/usr/bin/env python
# Filename: pre_yolov8_yaml.py 
"""
introduction: prepare yaml files for ultralytics.
            This script does not covert training data to YOLO format, so, before running this script,
            it is necessary to run yolov4_dir/pre_yolo_data.py

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 January, 2023
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as io_function
import parameters

def check_file_exist(f_path):
    if os.path.isfile(f_path) is False:
        raise IOError('%s does not exist, please run "yolov4_dir/pre_yolo_data.py first"')

def modify_data_yaml(data_dict, work_dir):
    # for key in data_dict.keys():
    #     print(key, data_dict[key])

    data_dict['path'] = work_dir
    data_dict['train'] = os.path.join('data','train.txt')
    check_file_exist(data_dict['train'])
    data_dict['val'] = os.path.join('data','val.txt')
    check_file_exist(data_dict['val'])

    name_file = os.path.join('data','obj.names')
    check_file_exist(name_file)
    name_list = io_function.read_list_from_txt(name_file)
    name_list = [item for item in name_list if len(item) > 0]   # remove empty lines
    names_dict = {}
    for idx, name in enumerate(name_list):
        names_dict[idx] = name
    data_dict['names'] = names_dict
    return data_dict


def modify_config_yaml(conf_dict, work_dir,para_file):

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    conf_dict['name'] = expr_name
    conf_dict['project'] = work_dir
    conf_dict['data'] = 'yolov8_data.yaml'

    process_num = parameters.get_digit_parameters(para_file,'process_num','int')
    conf_dict['workers'] = process_num

    network_setting_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    pre_trained_model = parameters.get_file_path_parameters(network_setting_ini,'pre_trained_model')
    io_function.is_file_exist(pre_trained_model)
    # model
    last_model = os.path.join(expr_name, 'weights', 'last.pt')
    if os.path.isfile(last_model):
        conf_dict['model'] = last_model
        conf_dict['resume'] = True  # resume
    else:
        conf_dict['resume'] = False
        conf_dict['model'] = pre_trained_model

    train_epoch_num = parameters.get_digit_parameters(network_setting_ini,'train_epoch_num','int')
    conf_dict['epochs'] = train_epoch_num

    return conf_dict


def create_yaml_files_from_yolov4_data(para_file):
    import yaml
    curr_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # data
    with open(os.path.join(script_dir, 'data.yaml'),'r') as f_obj:
        data_dict = yaml.safe_load(f_obj)
    n_data_dict = modify_data_yaml(data_dict, curr_dir)
    new_data_file = os.path.join(curr_dir, 'yolov8_data.yaml')
    with open(new_data_file, 'w') as f_obj:
        out = yaml.dump(n_data_dict, f_obj,sort_keys=False)
        print('save yaml file to %s' % new_data_file)

        # print(out)


    # configuration
    with open(os.path.join(script_dir, 'conf_yolov8.yaml'),'r') as f_obj:
        conf_dict = yaml.safe_load(f_obj)
    n_conf_dict = modify_config_yaml(conf_dict, curr_dir,para_file)
    new_conf_file = os.path.join(curr_dir, 'yolov8_conf.yaml')
    with open(new_conf_file, 'w') as f_obj:
        out = yaml.dump(n_conf_dict, f_obj, sort_keys=False)
        print('save yaml file to %s'%new_conf_file)


def main(options, args):
    para_file= args[0]
    create_yaml_files_from_yolov4_data(para_file)

if __name__ == '__main__':
    usage = "usage: %prog [options]  para_file"
    parser = OptionParser(usage=usage, version="1.0 2023-01-26")
    parser.description = 'Introduction: Prepare yaml files for ultralytics (YOLOv8) '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)