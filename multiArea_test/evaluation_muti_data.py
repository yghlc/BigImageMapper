#!/usr/bin/env python
# Filename: evaluation_muti_data 
"""
introduction: run evaluation for multiple data using multiple trained model.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 18 March, 2021
"""
import os, sys
from optparse import OptionParser

import time
from datetime import datetime

code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
sys.path.insert(0, code_dir)

machine_name = os.uname()[1]

import basic_src.io_function as io_function
import basic_src.basic as basic
import parameters

sys.path.insert(0, os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS'))
import slurm_utility

import GPUtil
from multiprocessing import Process

local_tasks = []

def modify_parameter(para_file, para_name, new_value):
    parameters.write_Parameters_file(para_file,para_name,new_value)

def run_exe_eval():
    job_sh = 'exe_eval.sh'
    res = os.system('./' + job_sh)
    if res != 0:
        sys.exit(1)
    return res


def run_evaluation_one_dataset(idx, area_ini,training_root_dir,template_dir):

    curr_dir = os.getcwd()

    run_eval_dir = os.path.basename(area_ini)[:-4] +'_%d'%idx
    main_para = 'main_para_eval_on_testData.ini'
    area_ini_name = os.path.basename(area_ini)

    if os.path.isdir(run_eval_dir) is False:
        io_function.mkdir(run_eval_dir)
        os.chdir(run_eval_dir)

        # copy and modify parameters
        io_function.copy_file_to_dst(os.path.join(template_dir, main_para), main_para)
        io_function.copy_file_to_dst(area_ini, area_ini_name)
        # set training_data_per=0, then all the data will be input for evaluation
        modify_parameter(main_para, 'training_regions', area_ini_name)
        io_function.copy_file_to_dst(os.path.join(template_dir, 'deeplabv3plus_xception65.ini'),'deeplabv3plus_xception65.ini')

        if 'login' in machine_name or 'shas' in machine_name or 'sgpu' in machine_name:
            io_function.copy_file_to_dst(os.path.join(template_dir, 'exe_curc.sh'), 'exe_curc.sh')
            io_function.copy_file_to_dst(os.path.join(template_dir, 'run_INsingularity_curc_GPU_tf.sh'),
                                         'run_INsingularity_curc_GPU_tf.sh')
            io_function.copy_file_to_dst(os.path.join(template_dir, 'job_tf_GPU.sh'), 'job_tf_GPU.sh')

            job_name = 'eval_%d_area' % idx
            slurm_utility.modify_slurm_job_sh('job_tf_GPU.sh', 'job-name', job_name)
        else:
            # copy
            io_function.copy_file_to_dst(os.path.join(template_dir, 'exe_eval.sh'), 'exe_eval.sh')

    else:
        os.chdir(run_eval_dir)

    # if run in curc cluster
    if 'login' in machine_name or 'shas' in machine_name or 'sgpu' in machine_name:

        while True:
            job_count = slurm_utility.get_submit_job_count(curc_username,job_name_substr='eval')
            if job_count >= max_run_jobs:
                print(machine_name, datetime.now(), 'You have submitted %d or more jobs, wait '%max_run_jobs)
                time.sleep(60)  #
                continue
            break

        # submit a job
        res = os.system('sbatch job_tf_GPU.sh')
        if res != 0:
            sys.exit(1)
    else:

        deviceIDs = []
        while True:
            # get available GPUs  # https://github.com/anderskm/gputil
            deviceIDs = GPUtil.getAvailable(order='memory', limit=100, maxLoad=0.5,
                                            maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
            basic.outputlogMessage('deviceIDs: %s'%str(deviceIDs))
            if len(deviceIDs) < 1:
                time.sleep(60)  # wait one minute, then check the available GPUs again
                continue
            break

        while True:
            job_count = basic.alive_process_count(local_tasks)
            if job_count >= max_run_jobs:
                print(machine_name, datetime.now(), '%d (>%d) jobs are running, wait '%(job_count,max_run_jobs))
                time.sleep(60)  #
                continue
            break

        job_sh = 'exe_eval.sh'
        gpuid = deviceIDs[0]
        # modify gpuid in exe_eval.sh
        with open(job_sh, 'r') as inputfile:
            list_of_all_the_lines = inputfile.readlines()
            for i in range(0, len(list_of_all_the_lines)):
                line = list_of_all_the_lines[i]
                if 'CUDA_VISIBLE_DEVICES' in line:
                    list_of_all_the_lines[i] = 'export CUDA_VISIBLE_DEVICES=%d\n'%gpuid
                    print('Set %s'%list_of_all_the_lines[i])
            # write the new file and overwrite the old one
        with open(job_sh,'w') as outputfile:
            outputfile.writelines(list_of_all_the_lines)
            outputfile.close()
        
        # run
        sub_process = Process(target=run_exe_eval)
        sub_process.start()
        local_tasks.append(sub_process)
        
        # wait until the assigned is used or exceed 100 seconds
        t0=time.time()
        while True:
            gpu_ids = GPUtil.getAvailable(order='memory', limit=100, maxLoad=0.5,
                                            maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
            t1 = time.time()
            # print(gpu_ids, t1-t0)
            if gpu_ids[0] != gpuid or (t1-t0) > 100:
                break 
            else:
                time.sleep(0.5)
            
        
        if sub_process.exitcode is not None and sub_process.exitcode !=0:
            sys.exit(1)



    os.chdir(curr_dir)

def main(options, args):
    # data_ini_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/WR_multiDate_inis')
    # training_root_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/ray_results/tune_dataAug_para_tesia')
    # template_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/eval_new_data')

    data_ini_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/area_multiDate_inis')
    training_root_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/multiArea_deeplabV3+_8')
    template_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/multiArea_deeplabV3+_8')

    if options.data_ini_dir_or_list is not None:
        data_ini_dir = options.data_ini_dir_or_list
    if options.training_root_dir is not None:
        training_root_dir = options.training_root_dir
    if options.template_dir is not None:
        template_dir  = options.template_dir

    # get data list
    if os.path.isdir(data_ini_dir):
        area_ini_list = io_function.get_file_list_by_ext('.ini',data_ini_dir,bsub_folder=False)
    else:
        area_ini_list = io_function.read_list_from_txt(data_ini_dir)
        # change to abslute path, since later, directory will be changed
        area_ini_list = [ os.path.abspath(item) for item in area_ini_list ]

    for idx,area_ini in enumerate(area_ini_list):
        basic.outputlogMessage('%d/%d evaluation on %d areas'%(idx, len(area_ini_list), idx))
        run_evaluation_one_dataset(idx,area_ini,training_root_dir,template_dir)


if __name__ == '__main__':
    usage = "usage: %prog [options] training_root_dir "
    parser = OptionParser(usage=usage, version="1.0 2021-03-17")
    parser.description = 'Introduction: collect parameters and training results (miou) '

    parser.add_option("-i", "--data_ini_dir_or_list",
                      action="store", dest="data_ini_dir_or_list",
                      help="the list for area ini file, if is folder, will get list from the folder")

    parser.add_option("-t", "--training_root_dir",
                      action="store", dest="training_root_dir",
                      help="the root directory of training, where the trained model is")

    parser.add_option("-m", "--template_dir",
                      action="store", dest="template_dir",  #default='multiArea_deeplabv3P_?????',
                      help="the folder where template are localted")

    parser.add_option("-j", "--max_run_jobs",
                      action="store", dest="max_run_jobs",type=int, default=5,
                      help="the maximum run jobs ")

    (options, args) = parser.parse_args()
    max_run_jobs = options.max_run_jobs
    curc_username = 'lihu9680'

    # parser.print_help()
    # # print(max_run_jobs)
    # sys.exit(1)

    main(options, args)