#!/usr/bin/env python
# Filename: deeplab_train 
"""
introduction: run the training and evaluation of deeplab

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 January, 2021
"""

import os, sys
from optparse import OptionParser
import math
import re

import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters

import basic_src.io_function as io_function
import basic_src.basic as basic

# pre-trained model with 21 classes
pre_trained_tar_21_classes = ['xception_65_coco_pretrained_2018_10_02.tar.gz',
                              'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
                              'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
                              'deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01.tar.gz',
                              'deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz',
                              'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
                              'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz']

pre_trained_tar_19_classes = ['deeplab_mnv3_large_cityscapes_trainfine_2019_11_15.tar.gz']

# the python with tensorflow 1.x installed
tf1x_python = 'python'

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
                  atrous_rates1,atrous_rates2,atrous_rates3,output_stride,crop_size_str,batch_size,iteration_num,depth_multiplier,
                  decoder_output_stride):

    # for more information, run: "python deeplab/train.py --help" or "python deeplab/train.py --helpfull"
    command_string = tf1x_python + ' ' \
        + train_script \
        + ' --logtostderr' \
        + ' --dataset='+dataset \
        + ' --num_classes='+str(num_of_classes) \
        + ' --train_split=%s '%train_split \
        + ' --base_learning_rate='+ str(base_learning_rate) \
        + ' --model_variant='+model_variant \
        + ' --train_crop_size='+crop_size_str \
        + ' --train_batch_size='+str(batch_size) \
        + ' --training_number_of_steps=' + str(iteration_num)\
        + ' --fine_tune_batch_norm=False' \
        + ' --tf_initial_checkpoint=' +init_checkpoint \
        + ' --train_logdir='+train_logdir \
        + ' --dataset_dir='+dataset_dir \
        + ' --num_clones=' + str(gpu_num)

    if output_stride is not None:
        command_string += ' --output_stride='+ str(output_stride)
    if decoder_output_stride is not None:
        command_string += ' --decoder_output_stride='+str(decoder_output_stride)
    if atrous_rates1 is not None:
        command_string += ' --atrous_rates=' + str(atrous_rates1)
    if atrous_rates2 is not None:
        command_string += ' --atrous_rates=' + str(atrous_rates2)
    if atrous_rates3 is not None:
        command_string += ' --atrous_rates=' + str(atrous_rates3)

    if depth_multiplier is not None:
        command_string += ' --depth_multiplier=' + str(depth_multiplier)

    if 'mobilenet_v3' in model_variant:
        # ' --image_pooling_crop_size = 769, 769 '
        command_string += ' --image_pooling_crop_size='+crop_size_str
        command_string += ' --image_pooling_stride=4,5 '
        command_string += ' --add_image_level_feature=1 '
        command_string += ' --aspp_convs_filters=128 '
        command_string += ' --aspp_with_concat_projection=0 '
        command_string += ' --aspp_with_squeeze_and_excitation=1 '
        command_string += ' --decoder_use_sum_merge=1 '
        command_string += ' --decoder_filters=19 '  # this is the same as number of class
        command_string += ' --decoder_output_is_logits=1 '
        command_string += ' --image_se_uses_qsigmoid=1 '

    res = os.system(command_string)
    if res != 0:
        sys.exit(res)



def evaluation_deeplab(evl_script,dataset, evl_split,num_of_classes, model_variant,
                       atrous_rates1,atrous_rates2,atrous_rates3,output_stride,train_logdir, evl_logdir,dataset_dir,
                       crop_size_str,max_eva_number,depth_multiplier,decoder_output_stride):

    # for information, run "python deeplab/eval.py  --helpfull"

# --eval_interval_secs: How often (in seconds) to run evaluation.
    #     (default: '300')
    #     (an integer)

    command_string = tf1x_python + ' ' \
                     + evl_script \
                     + ' --logtostderr' \
                     + ' --dataset='+dataset \
                     + ' --num_classes='+str(num_of_classes) \
                     + ' --eval_split=%s ' % evl_split \
                     + ' --model_variant=' + model_variant \
                     + ' --eval_crop_size='+crop_size_str \
                     + ' --checkpoint_dir=' + train_logdir \
                     + ' --eval_logdir=' + evl_logdir \
                     + ' --dataset_dir=' + dataset_dir \
                     + ' --max_number_of_evaluations=' + str(max_eva_number)

    if output_stride is not None:
        command_string += ' --output_stride='+ str(output_stride)
    if decoder_output_stride is not None:
        command_string += ' --decoder_output_stride='+str(decoder_output_stride)
    if atrous_rates1 is not None:
        command_string += ' --atrous_rates=' + str(atrous_rates1)
    if atrous_rates2 is not None:
        command_string += ' --atrous_rates=' + str(atrous_rates2)
    if atrous_rates3 is not None:
        command_string += ' --atrous_rates=' + str(atrous_rates3)

    if depth_multiplier is not None:
        command_string += ' --depth_multiplier=' + str(depth_multiplier)

    if 'mobilenet_v3' in model_variant:
        # ' --image_pooling_crop_size = 769, 769 '
        command_string += ' --image_pooling_crop_size='+crop_size_str
        command_string += ' --image_pooling_stride=4,5 '
        command_string += ' --add_image_level_feature=1 '
        command_string += ' --aspp_convs_filters=128 '
        command_string += ' --aspp_with_concat_projection=0 '
        command_string += ' --aspp_with_squeeze_and_excitation=1 '
        command_string += ' --decoder_use_sum_merge=1 '
        command_string += ' --decoder_filters=19 '  # this is the same as number of class
        command_string += ' --decoder_output_is_logits=1 '
        command_string += ' --image_se_uses_qsigmoid=1 '

    res = os.system(command_string)
    if res != 0:
        sys.exit(res)

def get_miou_list_class_all(train_log_dir,class_num):

    # add the tensorboard in the tf1x version
    tf1x_dir = os.path.join(os.path.dirname(os.path.dirname(tf1x_python)),'lib','python3.7','site-packages')
    sys.path.insert(0, tf1x_dir)

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Loading too much data is slow...
    # tf_size_guidance on how much data the EventAccumulator should
    #  |          store in memory. The DEFAULT_SIZE_GUIDANCE tries not to store too much
    #  |          so as to avoid OOMing the client. The size_guidance should be a map
    #  |          from a `tagType` string to an integer representing the number of
    #  |          items to keep per tag for items of that `tagType`. If the size is 0,
    #  |          all events are stored.
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 0,       # set a 0, to load all scalars
        'histograms': 1
    }

    event_acc = EventAccumulator(train_log_dir, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    # tag_dict = event_acc.Tags()
    # io_function.save_dict_to_txt_json('event_acc.txt',tag_dict)

    miou_dic = {}
    for class_id in range(class_num):
        name  = 'class_%d'%class_id
        miou_class_event = event_acc.Scalars('eval/miou_1.0_'+name)
        miou_class_list = [item[2] for item in miou_class_event ]  # item[0] is wall_time, item[1] is step, item [2] is the value
        miou_dic[name] = miou_class_list

    miou_class_overall = event_acc.Scalars('eval/miou_1.0_overall')
    miou_class_list = [item[2] for item in miou_class_overall]
    miou_dic['overall'] = miou_class_list
    io_function.save_dict_to_txt_json('miou.txt', miou_dic)

    return miou_dic


def get_trained_iteration(TRAIN_LOGDIR):
    checkpoint = os.path.join(TRAIN_LOGDIR,'checkpoint')
    if os.path.isfile(checkpoint):
        with open(checkpoint,'r') as f_obj:
            lines = f_obj.readlines()
            strings = re.findall(r'ckpt-\d+',lines[0])
            if len(strings) < 0:
                return 0
            else:
                return int(strings[0][5:])
    else:
        return 0

def train_evaluation_deeplab(WORK_DIR,deeplab_dir,expr_name, para_file, network_setting_ini,gpu_num):

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
    batch_size = parameters.get_digit_parameters(network_setting_ini,'batch_size','int')
    # maximum iteration number
    iteration_num = parameters.get_digit_parameters(network_setting_ini,'iteration_num','int')
    base_learning_rate = parameters.get_digit_parameters(network_setting_ini,'base_learning_rate','float')

    train_output_stride = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'train_output_stride','int')
    train_atrous_rates1 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'train_atrous_rates1','int')
    train_atrous_rates2 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'train_atrous_rates2','int')
    train_atrous_rates3 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'train_atrous_rates3','int')

    inf_output_stride = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'inf_output_stride','int')
    inf_atrous_rates1 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'inf_atrous_rates1','int')
    inf_atrous_rates2 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'inf_atrous_rates2','int')
    inf_atrous_rates3 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'inf_atrous_rates3','int')

    # depth_multiplier default is 1.0.
    depth_multiplier = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'depth_multiplier', 'float')

    decoder_output_stride = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'decoder_output_stride', 'int')

    train_script = os.path.join(deeplab_dir, 'train.py')
    train_split = os.path.splitext(parameters.get_string_parameters(para_file,'training_sample_list_txt'))[0]
    model_variant = parameters.get_string_parameters(network_setting_ini, 'model_variant')
    checkpoint = parameters.get_string_parameters(network_setting_ini, 'tf_initial_checkpoint')
    init_checkpoint_files = io_function.get_file_list_by_pattern(INIT_FOLDER, checkpoint + '*')
    if len(init_checkpoint_files) < 1:
        raise IOError('No initial checkpoint in %s with pattern: %s'%(INIT_FOLDER,checkpoint))
    init_checkpoint = os.path.join(INIT_FOLDER,checkpoint)
    dataset = parameters.get_string_parameters(para_file,'dataset_name')
    num_classes_noBG = parameters.get_digit_parameters_None_if_absence(para_file, 'NUM_CLASSES_noBG', 'int')
    assert num_classes_noBG != None
    if pre_trained_tar in pre_trained_tar_21_classes:
        print('warning, pretrained model %s is trained with 21 classes, set num_of_classes to 21'%pre_trained_tar)
        num_classes_noBG = 20
    if pre_trained_tar in pre_trained_tar_19_classes:
        print('warning, pretrained model %s is trained with 19 classes, set num_of_classes to 19'%pre_trained_tar)
        num_classes_noBG = 18
    num_of_classes = num_classes_noBG + 1

    image_crop_size = parameters.get_string_list_parameters(para_file, 'image_crop_size')
    if len(image_crop_size) != 2 and image_crop_size[0].isdigit() and image_crop_size[1].isdigit():
        raise ValueError('image_crop_size should be height,width')
    crop_size_str = ','.join(image_crop_size)

    evl_script = os.path.join(deeplab_dir, 'eval.py')
    evl_split = os.path.splitext(parameters.get_string_parameters(para_file,'validation_sample_list_txt'))[0]
    max_eva_number = 1

    b_early_stopping = parameters.get_bool_parameters(para_file,'b_early_stopping')

    # validation interval (epoch)
    validation_interval = parameters.get_digit_parameters_None_if_absence(para_file,'validation_interval','int')
    train_count, val_count = get_train_val_sample_count(WORK_DIR, para_file)
    iter_per_epoch = math.ceil(train_count/batch_size)
    total_epoches = math.ceil(iteration_num/iter_per_epoch)
    already_trained_iteration  = get_trained_iteration(TRAIN_LOGDIR)
    if already_trained_iteration >= iteration_num:
        basic.outputlogMessage('Training already run %d iterations, skip'%already_trained_iteration)
        return True
    if validation_interval is None:
        basic.outputlogMessage('No input validation_interval, so training to %d, then evaluating in the end'%iteration_num)
        # run training
        train_deeplab(train_script,dataset, train_split,num_of_classes, base_learning_rate, model_variant, init_checkpoint, TRAIN_LOGDIR,
                      dataset_dir, gpu_num,
                      train_atrous_rates1, train_atrous_rates2, train_atrous_rates3, train_output_stride,crop_size_str, batch_size,iteration_num,
                      depth_multiplier,decoder_output_stride)

        # run evaluation
        evaluation_deeplab(evl_script,dataset, evl_split, num_of_classes,model_variant,
                           inf_atrous_rates1,inf_atrous_rates2,inf_atrous_rates3,inf_output_stride,TRAIN_LOGDIR, EVAL_LOGDIR,
                           dataset_dir,crop_size_str, max_eva_number,depth_multiplier,decoder_output_stride)
    else:
        basic.outputlogMessage('training to the maximum iteration of %d, and evaluating very %d epoch(es)' % (iteration_num,validation_interval))
        for epoch in range(1, total_epoches + validation_interval, validation_interval):

            to_iter_num = min(epoch*iter_per_epoch, iteration_num)
            if to_iter_num <= already_trained_iteration:
                continue
            basic.outputlogMessage('training and evaluating to %d epoches (to iteration: %d)' % (epoch,to_iter_num))

            # run training
            train_deeplab(train_script, dataset, train_split, num_of_classes, base_learning_rate, model_variant,
                          init_checkpoint, TRAIN_LOGDIR,
                          dataset_dir, gpu_num,
                          train_atrous_rates1, train_atrous_rates2, train_atrous_rates3, train_output_stride,crop_size_str,
                          batch_size, to_iter_num,depth_multiplier,decoder_output_stride)

            # run evaluation
            evaluation_deeplab(evl_script, dataset, evl_split, num_of_classes, model_variant,
                               inf_atrous_rates1, inf_atrous_rates2, inf_atrous_rates3, inf_output_stride, TRAIN_LOGDIR, EVAL_LOGDIR,
                               dataset_dir,crop_size_str, max_eva_number,depth_multiplier,decoder_output_stride)

            # get miou
            miou_dict = get_miou_list_class_all(EVAL_LOGDIR, num_of_classes)
            # check if need to early stopping
            if b_early_stopping:
                if len(miou_dict['overall']) >= 5:
                    # if the last five miou did not improve, then stop training
                    if np.all(np.diff(miou_dict['overall'][-5:]) < 0.0001):
                        basic.outputlogMessage('early stopping: stop training because overall miou did not improved in the last five evaluation')
                        break


# def init_for_test_function():
#     code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
#     sys.path.insert(0, code_dir)
#     import parameters
#     global parameters   # use global to make parameters be visiable in this file.
#     import basic_src.io_function as io_function
#     global io_function
#     import basic_src.basic as basic
#     global basic
#
#     global work_dir
#     work_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL/working_dir')



def main(options, args):

    print("%s : train deeplab" % os.path.basename(sys.argv[0]))

    para_file = sys.argv[1]
    gpu_num = int(sys.argv[2])
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    network_setting_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    tf_research_dir = parameters.get_directory_None_if_absence(network_setting_ini, 'tf_research_dir')
    print(tf_research_dir)
    if tf_research_dir is None:
        raise ValueError('tf_research_dir is not in %s' % para_file)
    if os.path.isdir(tf_research_dir) is False:
        raise ValueError('%s does not exist' % tf_research_dir)
    # sys.path.insert(0, tf_research_dir)
    # sys.path.insert(0, os.path.join(tf_research_dir,'slim'))
    # print(sys.path)
    # need to change PYTHONPATH, otherwise, deeplab cannot be found
    if os.getenv('PYTHONPATH'):
        os.environ['PYTHONPATH'] = os.getenv('PYTHONPATH') + ':' + tf_research_dir + ':' + os.path.join(tf_research_dir,
                                                                                                        'slim')
    else:
        os.environ['PYTHONPATH'] = tf_research_dir + ':' + os.path.join(tf_research_dir, 'slim')
    # os.system('echo $PYTHONPATH ')

    global tf1x_python
    tf1x_python = parameters.get_file_path_parameters(network_setting_ini,'tf1x_python')

    deeplab_dir = os.path.join(tf_research_dir, 'deeplab')
    WORK_DIR = os.getcwd()

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')


    train_evaluation_deeplab(WORK_DIR, deeplab_dir, expr_name, para_file, network_setting_ini,gpu_num)


if __name__ == '__main__':

    usage = "usage: %prog [options] para_file gpu_num"
    parser = OptionParser(usage=usage, version="1.0 2021-01-19")
    parser.description = 'Introduction: training and evaluating of Deeplab '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)










