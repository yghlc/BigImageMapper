#!/usr/bin/env python
# Filename: deeplab_train 
"""
introduction: run the training of deeplab

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 January, 2021
"""

import os, sys
import math

def get_train_val_sample_count(work_dir, para_file):

    train_sample_txt = parameters.get_string_parameters(para_file, 'training_sample_list_txt')
    val_sample_txt = parameters.get_string_parameters(para_file, 'validation_sample_list_txt')
    train_list_txt = os.path.join(work_dir,'list', train_sample_txt)
    val_list_txt = os.path.join(work_dir, 'list', val_sample_txt)

    train_lines = io_function.read_list_from_txt(train_list_txt)
    val_lines = io_function.read_list_from_txt(val_list_txt)
    basic.outputlogMessage('The count of training and validation samples are %d and %d'%(len(train_lines), len(val_lines)))

    return len(train_lines), len(val_lines)

def train_deeplab(train_script,dataset,train_split,num_of_classes,base_learning_rate,model_variant, init_checkpoint,train_logdir,dataset_dir, gpu_num,
                  atrous_rates1,atrous_rates2,atrous_rates3,output_stride,batch_size,iteration_num):

    # for more information, run: "python deeplab/train.py --help" or "python deeplab/train.py --helpfull"
    command_string = 'python ' \
        + train_script \
        + ' --logtostderr' \
        + ' --dataset='+dataset \
        + ' --num_classes='+str(num_of_classes) \
        + ' --train_split=%s '%train_split \
        + ' --base_learning_rate='+ str(base_learning_rate) \
        + ' --model_variant='+model_variant \
        + ' --atrous_rates='+str(atrous_rates1) \
        + ' --atrous_rates='+str(atrous_rates2) \
        + ' --atrous_rates='+str(atrous_rates3) \
        + ' --output_stride='+ str(output_stride) \
        + ' --decoder_output_stride=4 ' \
        + ' --train_crop_size=513,513' \
        + ' --train_batch_size='+str(batch_size) \
        + ' --training_number_of_steps=' + str(iteration_num)\
        + ' --fine_tune_batch_norm=False' \
        + ' --tf_initial_checkpoint=' +init_checkpoint \
        + ' --train_logdir='+train_logdir \
        + ' --dataset_dir='+dataset_dir \
        + ' --num_clones=' + str(gpu_num)

    res = os.system(command_string)
    if res != 0:
        sys.exit(res)



def evaluation_deeplab(evl_script,dataset, evl_split,num_of_classes, model_variant,
                       atrous_rates1,atrous_rates2,atrous_rates3,output_stride,train_logdir, evl_logdir,dataset_dir, max_eva_number):

    # for information, run "python deeplab/eval.py  --helpfull"

# --eval_interval_secs: How often (in seconds) to run evaluation.
    #     (default: '300')
    #     (an integer)

    command_string = 'python ' \
                     + evl_script \
                     + ' --logtostderr' \
                     + ' --dataset='+dataset \
                     + ' --num_classes='+str(num_of_classes) \
                     + ' --eval_split=%s ' % evl_split \
                     + ' --model_variant=' + model_variant \
                     + ' --atrous_rates=' + str(atrous_rates1) \
                     + ' --atrous_rates=' + str(atrous_rates2) \
                     + ' --atrous_rates=' + str(atrous_rates3) \
                     + ' --output_stride=' + str(output_stride) \
                     + ' --decoder_output_stride=4 ' \
                     + ' --eval_crop_size=513,513' \
                     + ' --checkpoint_dir=' + train_logdir \
                     + ' --eval_logdir=' + evl_logdir \
                     + ' --dataset_dir=' + dataset_dir \
                     + ' --max_number_of_evaluations=' + str(max_eva_number)

    res = os.system(command_string)
    if res != 0:
        sys.exit(res)

def train_evaluation_deeplab(WORK_DIR,deeplab_dir,expr_name, para_file, network_setting_ini):

    # prepare training folder
    EXP_FOLDER = expr_name
    INIT_FOLDER =  os.path.join(WORK_DIR,EXP_FOLDER,'init_models')
    TRAIN_LOGDIR = os.path.join(WORK_DIR,EXP_FOLDER,'train')
    EVAL_LOGDIR = os.path.join(WORK_DIR,EXP_FOLDER,'eval')
    VIS_LOGDIR = os.path.join(WORK_DIR,EXP_FOLDER,'vis')
    EXPORT_DIR = os.path.join(WORK_DIR,EXP_FOLDER,'export')

    io_function.mkdir(INIT_FOLDER)
    io_function.mkdir(TRAIN_LOGDIR)
    io_function.mkdir(EVAL_LOGDIR)
    io_function.mkdir(VIS_LOGDIR)
    io_function.mkdir(EXPORT_DIR)

    # prepare the tensorflow check point (pretrained model) for training
    pre_trained_dir = parameters.get_directory_None_if_absence(network_setting_ini, 'pre_trained_model_folder')
    pre_trained_tar = parameters.get_string_parameters(network_setting_ini, 'TF_INIT_CKPT')
    pre_trained_path = os.path.join(pre_trained_dir, pre_trained_tar)
    if os.path.isfile(pre_trained_path) is False:
        print('pre-trained model: %s not exist, try to download'%pre_trained_path)
        # try to download the file
        pre_trained_url = parameters.get_string_parameters_None_if_absence(network_setting_ini, 'pre_trained_model_url')
        res  = os.system('wget %s '%pre_trained_url)
        if res != 0:
            sys.exit(res)
        io_function.movefiletodir(pre_trained_tar,pre_trained_dir)

    # unpack pre-trained model to INIT_FOLDER
    os.chdir(INIT_FOLDER)
    res = os.system('tar -xf %s'% pre_trained_path)
    if res != 0:
        raise IOError('failed to unpack %s'%pre_trained_path)
    os.chdir(WORK_DIR)

    dataset_dir = os.path.join(WORK_DIR, 'tfrecord')
    batch_size = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'batch_size','int')
    # maximum iteration number
    iteration_num = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'iteration_num','int')
    base_learning_rate = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'base_learning_rate','float')

    output_stride = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'output_stride','int')
    atrous_rates1 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'atrous_rates1','int')
    atrous_rates2 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'atrous_rates2','int')
    atrous_rates3 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'atrous_rates3','int')

    train_script = os.path.join(deeplab_dir, 'train.py')
    train_split = os.path.splitext(parameters.get_string_parameters(para_file,'training_sample_list_txt'))[0]
    model_variant = parameters.get_string_parameters(network_setting_ini, 'model_variant')
    checkpoint = parameters.get_string_parameters(network_setting_ini, 'tf_initial_checkpoint')
    init_checkpoint = os.path.join(INIT_FOLDER,checkpoint)
    dataset = parameters.get_string_parameters(para_file,'dataset_name')
    num_classes_noBG = parameters.get_digit_parameters_None_if_absence(para_file, 'NUM_CLASSES_noBG', 'int')
    assert num_classes_noBG != None
    num_of_classes = num_classes_noBG + 1

    evl_script = os.path.join(deeplab_dir, 'eval.py')
    evl_split = os.path.splitext(parameters.get_string_parameters(para_file,'validation_sample_list_txt'))[0]
    max_eva_number = 1

    # validation interval (epoch)
    validation_interval = parameters.get_digit_parameters_None_if_absence(para_file,'validation_interval','int')
    train_count, val_count = get_train_val_sample_count(WORK_DIR, para_file)
    iter_per_epoch = math.ceil(train_count/batch_size)
    total_epoches = math.ceil(iteration_num/iter_per_epoch)
    if validation_interval is None:
        basic.outputlogMessage('No input validation_interval, so training to %d, then evaluating in the end'%iteration_num)
        # run training
        train_deeplab(train_script,dataset, train_split,num_of_classes, base_learning_rate, model_variant, init_checkpoint, TRAIN_LOGDIR,
                      dataset_dir, gpu_num,
                      atrous_rates1, atrous_rates2, atrous_rates3, output_stride, batch_size, iteration_num)

        # run evaluation
        evaluation_deeplab(evl_script,dataset, evl_split, num_of_classes,model_variant,
                           atrous_rates1,atrous_rates2,atrous_rates3,output_stride,TRAIN_LOGDIR, EVAL_LOGDIR, dataset_dir, max_eva_number)
    else:
        basic.outputlogMessage('training to %d, and evaluating very %d epoch(es)' % (iteration_num,validation_interval))
        for epoch in range(1, total_epoches + iter_per_epoch,iter_per_epoch):

            to_iter_num = min(epoch*iter_per_epoch, iteration_num)
            basic.outputlogMessage('training and evaluating to %d epoches (to iteration: %d)' % (epoch,to_iter_num))

            # run training
            train_deeplab(train_script, dataset, train_split, num_of_classes, base_learning_rate, model_variant,
                          init_checkpoint, TRAIN_LOGDIR,
                          dataset_dir, gpu_num,
                          atrous_rates1, atrous_rates2, atrous_rates3, output_stride, batch_size, to_iter_num)

            # run evaluation
            evaluation_deeplab(evl_script, dataset, evl_split, num_of_classes, model_variant,
                               atrous_rates1, atrous_rates2, atrous_rates3, output_stride, TRAIN_LOGDIR, EVAL_LOGDIR,
                               dataset_dir, max_eva_number)


if __name__ == '__main__':
    print("%s : train deeplab" % os.path.basename(sys.argv[0]))

    para_file = sys.argv[1]
    gpu_num = int(sys.argv[2])
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
    sys.path.insert(0, code_dir)
    import parameters

    import basic_src.io_function as io_function
    import basic_src.basic as basic


    tf_research_dir = parameters.get_directory_None_if_absence(para_file,'tf_research_dir')
    print(tf_research_dir)
    if tf_research_dir is None:
        raise ValueError('tf_research_dir is not in %s'%para_file)
    if os.path.isdir(tf_research_dir) is False:
        raise ValueError('%s does not exist' % tf_research_dir)
    # sys.path.insert(0, tf_research_dir)
    # sys.path.insert(0, os.path.join(tf_research_dir,'slim'))
    # print(sys.path)
    # need to change PYTHONPATH, otherwise, deeplab cannot be found
    if os.getenv('PYTHONPATH'):
        os.environ['PYTHONPATH'] = os.getenv('PYTHONPATH') + ':' + tf_research_dir + ':' +os.path.join(tf_research_dir,'slim')
    else:
        os.environ['PYTHONPATH'] = tf_research_dir + ':' +os.path.join(tf_research_dir,'slim')
    # os.system('echo $PYTHONPATH ')

    deeplab_dir = os.path.join(tf_research_dir,'deeplab')
    WORK_DIR = os.getcwd()

    expr_name = parameters.get_string_parameters(para_file,'expr_name')
    network_setting_ini = parameters.get_string_parameters(para_file,'network_setting_ini')

    train_evaluation_deeplab(WORK_DIR, deeplab_dir, expr_name, para_file, network_setting_ini)








