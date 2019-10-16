#!/usr/bin/env python
# Filename: copyTolocal_postPro 
"""
introduction: copy prediction results from chpc cluster, then conduct post-processing


authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 October, 2019
"""

import os,sys
import time

HOME = os.path.expanduser('~')
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.basic as basic
import basic_src.io_function as io_function
import parameters

import psutil
import getpass  # for get current username
import random

server="s1155090023@chpc-login01.itsc.cuhk.edu.hk"
remote_workdir='/users/s1155090023/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2/autoMapping'
# test
# server="hlc@10.0.0.203"
# remote_workdir='/home/hlc/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2/autoMapping'

run_folder=os.path.join(remote_workdir, 'QTP_deeplabV3+_3')
outdir = 'multi_inf_results'

def copy_remote_file_to_local(re_path,local_dir='./'):
    # os.system('scp '+ server + ':'+re_path + ' ' + local_dir)
    cmd_str = 'scp '+ server + ':'+re_path + ' ' + local_dir
    basic.exec_command_string(cmd_str)

def copy_remote_dir_to_local(re_path,local_dir='./'):
    # if os.path.isdir(local_dir):
    local_dir = os.path.dirname(local_dir)
    # os.system('scp -r '+ server + ':'+re_path + ' ' + local_dir)
    # rsync options, r for folder, a: archive mode, -z, compress file data during the transfer
    cmd_str = 'rsync -rz ' + server + ':' + re_path + ' ' + local_dir
    basic.exec_command_string(cmd_str)

def get_remote_file_list(pattern):
    command = 'ssh ' + server + ' ls ' + pattern
    result = basic.exec_command_string_output_string(command)
    # result = os.system(command)
    # print('result',type(result))

    if "No such file or directory" in result:
        basic.outputlogMessage(result)
        return False
    else:
        file_list = result.split('\n')
        return file_list

def is_file_exist_in_folder(folder):
    # only check the first ten files
    for i in range(10):
        if os.path.isfile(os.path.join(folder,'I0_%d.tif'%i)):
            return True
    return False
    # file_list = io_function.get_file_list_by_pattern(folder, '*.*')  # this may take time if a lot of file exist
    # if len(file_list) > 0:
    #     return True
    # else:
    #     return False

    pass


def outputlogMessage(message):
    basic.outputlogMessage(message)

def is_gdal_merge_running():
    # process_names = [proc.name() for proc in psutil.process_iter()]
    user = getpass.getuser()
    for proc in psutil.process_iter():
        if 'python' in proc.name() and proc.username() == user:
            cmd_args_list = proc.cmdline()[:20] # only get first 20 arguments (include python)
            if 'gdal_merge.py' in cmd_args_list[1]:
                return True
    return False



if __name__ == "__main__":
    # copy the inf_image_list.txt to local
    copy_remote_file_to_local(os.path.join(run_folder, 'inf_image_list.txt'))
    copy_remote_file_to_local(os.path.join(run_folder, 'para_qtp.ini'))
    os.system('sed -i -e  s%/users/s1155090023%/home/hlc%g para_qtp.ini')  # change to local path

    with open('inf_image_list.txt', 'r') as inf_obj:
        inf_img_list = [name.strip() for name in inf_obj.readlines()]

    img_count = len(inf_img_list)
    if img_count < 1:
        raise ValueError('No image in inf_image_list.txt')

    done_list = []  # a list of files, e.g., 15.txt_done, which incate the task is complete
    # check multi_inf_results exist, not necessary to remove it, since scp can replace files inside

    os.system('mkdir -p ' + outdir)

    para_file = sys.argv[1]

    # expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
    # NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
    # trail=iter${NUM_ITERATIONS}
    #
    # testid=$(basename $PWD)_${expr_name}_${trail}
    # output=${testid}.tif

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    NUM_ITERATIONS = parameters.get_string_parameters(para_file, 'export_iteration_num')
    trail = 'iter' + NUM_ITERATIONS
    testid = os.path.basename(os.getcwd()) + '_' + expr_name + '_' + trail
    output = testid + '.tif'

    while len(done_list) < img_count:
        re_file_list = get_remote_file_list(
            os.path.join(run_folder, outdir, '*.txt_done'))  # + '/multi_inf_results/*.txt_done')
        if re_file_list is False:
            wait_sec = random.randint(60, 120)
            basic.outputlogMessage('No completed prediction sub-images, wait %d seconds, PID: %d '% (wait_sec,os.getpid()))
            time.sleep(wait_sec)  # wait one minute
            continue

        remote_done_count = len(re_file_list)
        if len(done_list) == remote_done_count:
            wait_sec = random.randint(60, 120)
            basic.outputlogMessage('No completed prediction sub-images, wait %d seconds, PID: %d '% (wait_sec,os.getpid()))
            time.sleep(wait_sec)  # wait one minute
            continue

        for re_task_file in re_file_list:

            base_name = os.path.basename(re_task_file)
            task_id = base_name.split('.')[0]
            done_task = os.path.join(outdir, base_name)
            if os.path.isfile(done_task):
                if base_name not in done_list:
                    done_list.append(base_name)
                continue

            # copy the remote folder
            time0 = time.time()
            re_task_folder = os.path.join(os.path.dirname(re_task_file), 'I' + task_id)
            local_folder = os.path.join(outdir, 'I' + task_id)
            # if it already exist, then skip to next
            if os.path.isdir(local_folder):# and is_file_exist_in_folder(local_folder):
                if base_name not in done_list:
                    done_list.append(base_name)
                basic.outputlogMessage('folder %s is being processing by other, skip' % local_folder)
                continue
            os.system('mkdir -p '+ local_folder)   # create folder immediately to indicate this task is being handling
            copy_remote_dir_to_local(re_task_folder, local_folder)
            basic.outputlogMessage('copying folder %s cost %.2f seconds' % (local_folder, (time.time() - time0)))

            time0 = time.time()
            cwd_dir = os.getcwd()
            os.chdir(local_folder)
            # gdal_merge.py, which is time-consuming
            task_out_tif = 'I' + task_id + '_' + output
            # # only allow one gdal_mergy.py is running on a machine (not helpful )
            # while is_gdal_merge_running() is True:
            #     time.sleep(3)   # wait three seconds

            if os.path.isfile(task_out_tif) is False:
                cmd_str = 'gdal_merge.py -init 0 -n 0 -a_nodata 0 -o ' + task_out_tif + ' ' + ' I0_*.tif'
                if basic.exec_command_string_one_file(cmd_str, task_out_tif) is False:
                    raise IOError('error, failed to generate %s' % os.path.abspath(task_out_tif))

            # gdal_polygonize.py
            task_out_shp = 'I' + task_id + '_' + testid + '.shp'
            if os.path.isfile(task_out_shp) is False:
                # os.system('gdal_polygonize.py -8 '+ task_out_tif +  ' -b 1 -f "ESRI Shapefile" ' + task_out_shp)
                cmd_str = 'gdal_polygonize.py -8 ' + task_out_tif + ' -b 1 -f "ESRI Shapefile" ' + task_out_shp
                if basic.exec_command_string_one_file(cmd_str, task_out_shp) is False:
                    raise IOError('error, failed to generate %s' % os.path.abspath(task_out_shp))

            os.chdir(cwd_dir)
            basic.outputlogMessage(
                'merging and polygonizing of %s cost %.2f seconds' % (local_folder, (time.time() - time0)))

            # indicating it is done
            copy_remote_file_to_local(re_task_file, os.path.join(outdir, base_name))
            done_list.append(base_name)

    # after all done,  run exe_qtp.sh for further post-processing and merging of shape file
    # os.system('./exe_qtp.sh') # use this script in "post_pro_chpc.sh" on Cryo03