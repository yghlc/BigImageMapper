#!/usr/bin/env python
# Filename: test_kfold_cross_val_multi 
"""
introduction: run tests on k-fold cross validation, especially for multiple shape file (multi-temporal training polygons)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 March, 2020
"""

import os,sys
from optparse import OptionParser
from datetime import datetime

# added path of DeeplabforRS
deeplabRS = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabRS)

get_training_polygons_script = os.path.join(deeplabRS,'get_training_polygons.py')

import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic

log = "time_cost.txt"

# remember to modify this if it has change
trained_model_dir = 'exp6'      # read from para_file

def print2file(save_file, string):
    print(string)
    with open(save_file,'a') as f_obj:
        f_obj.writelines(string + '\n')

def create_shp_subset_polygons(dir_sub, training_shpAll, file_name, k_value):

    curr_dir = os.getcwd()

    # change the running folder
    os.chdir(dir_sub)
    args_list = [get_training_polygons_script, training_shpAll, file_name, '-k', str(k_value)]
    basic.exec_command_args_list(args_list)
    # change back
    os.chdir(curr_dir)

    return True

def repalce_string_in_file(txt_path, old_str, new_str):

    import fileinput
    for line in fileinput.input(txt_path, inplace=True):
        # inside this loop the STDOUT will be redirected to the file
        # the comma after each print statement is needed to avoid double line breaks
        print(line.replace(old_str, new_str), end="")

def train_kfold_cross_val(multi_training_files_allPolygons, multi_training_files, k_value, test_num):

    ##################################################################
    # get subset of polygons
    training_shp_all = []
    with open(multi_training_files_allPolygons,'r') as f_obj:
        training_lines = f_obj.readlines()
        for line in training_lines:
            line = line.strip()
            training_shp_all.append(line.split(':')[-1])        # the last one is the shape file

    for training_shpAll in training_shp_all:

        dir = os.path.dirname(training_shpAll)
        file_name = os.path.basename(training_shpAll)
        dir_sub = os.path.join(dir,'%d-fold_cross_val_t%d'%(k_value,test_num))

        if os.path.isdir(dir_sub) is False:

            # will save to dir_sub}
            io_function.mkdir(dir_sub)
            create_shp_subset_polygons(dir_sub, training_shpAll, file_name, k_value)
        else:
            # check shape file existence
            sub_shps = io_function.get_file_list_by_pattern(dir_sub,'*.shp')
            if len(sub_shps) == k_value:
                print2file(log, "subset of shapefile already exist, skip creating new" )
            else:
                create_shp_subset_polygons(dir_sub, training_shpAll, file_name, k_value)


    ##################################################################
    # training on k subset
    for idx in range(1, k_value+1):
        # remove previous trained model (the setting are the same to exp10)
        if os.path.isdir(trained_model_dir):
            io_function.delete_file_or_dir(trained_model_dir)

        print2file(log,"run training and inference of the %d_th fold"%idx )

        # replace shape file path in "multi_training_files"

        io_function.copy_file_to_dst(multi_training_files_allPolygons,multi_training_files,overwrite=True)
        # replace shape file path in multi_training_files
        for training_shpAll in training_shp_all:
            dir = os.path.dirname(training_shpAll)
            file_name_no_ext = os.path.splitext(os.path.basename(training_shpAll))[0]
            dir_sub = os.path.join(dir, '%d-fold_cross_val_t%d' % (k_value, test_num))

            new_shp_path = os.path.join(dir_sub,'%s_%dfold_%d.shp'%(file_name_no_ext,k_value,idx))
            repalce_string_in_file(multi_training_files,training_shpAll,new_shp_path)

        # modify exe.sh
        io_function.copy_file_to_dst('exe_template_kfold.sh', 'exe_qtp.sh', overwrite=True)
        new_line = '%dfold_%d_t%d'%(k_value,idx,test_num)
        repalce_string_in_file('exe_qtp.sh', 'x_test_num', new_line)

        # check results existence
        result_shp = io_function.get_file_list_by_pattern('result_backup','*'+new_line+'*/*.shp')
        if len(result_shp) > 0:
            print2file(log,"results of test: %s already exist, skip"%new_line)
        else:
            # run training
            print2file(log,"start: test:%d the %d_th fold"%(test_num,idx))
            argslist = ['./exe_qtp.sh']
            return_code = basic.exec_command_args_list(argslist)
            # exit code is not 0, means something wrong, then quit
            if return_code != 0:
                sys.exit(return_code)

    pass

def main(options, args):

    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print2file(log,time_str)

    para_file = options.para_file
    k_value = int(args[0])
    test_num = int(args[1])

    print2file(log,'kvalue : %d'%k_value)
    print2file(log,'test_num : %d'%test_num)

    global trained_model_dir
    trained_model_dir = parameters.get_string_parameters(para_file,'expr_name')

    # get the path of multi training polygons
    multi_training_files = parameters.get_string_parameters_None_if_absence(para_file,'multi_training_files')
    if multi_training_files is None:
        raise ValueError('multi_training_files is not set in the %s' % para_file)

    io_function.is_file_exist(multi_training_files)

    # backup the original training file which contains the full set of polyogns
    training_files_allPolygons = io_function.get_name_by_adding_tail(multi_training_files,'allPolygons')
    if os.path.isfile(training_files_allPolygons) is False:
        io_function.copy_file_to_dst(multi_training_files, training_files_allPolygons)
    else:
        basic.outputlogMessage('The full set polygons already exist') #%multi_training_files

    # training on using the k subset
    train_kfold_cross_val(training_files_allPolygons, multi_training_files ,k_value,test_num)

    # remove the backup of multi_training_files
    # io_function.copy_file_to_dst(training_files_allPolygons,multi_training_files)
    # io_function.delete_file_or_dir(training_files_allPolygons)



if __name__ == "__main__":

    usage = "usage: %prog [options] k_value test_number "
    parser = OptionParser(usage=usage, version="1.0 2020-03-08")
    parser.description = 'Introduction: run k-fold cross-validation '

    parser.add_option("-p", "--para",
                      action="store", dest="para_file",
                      help="the parameters file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    # set parameters files
    if options.para_file is None:
        print('error, no parameters file')
        parser.print_help()
        sys.exit(2)
    else:
        parameters.set_saved_parafile_path(options.para_file)

    main(options, args)

