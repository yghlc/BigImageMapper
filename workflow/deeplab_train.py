#!/usr/bin/env python
# Filename: deeplab_train 
"""
introduction: run the training of deeplab

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 January, 2021
"""

import os, sys


def train_deeplab(train_script,train_split,base_learning_rate,model_variant, init_checkpoint,train_logdir,dataset_dir, gpu_num,
                  atrous_rates1,atrous_rates2,atrous_rates3,output_stride,batch_size,iteration_num):

    # for more information, run: "python deeplab/train.py --help" or "python deeplab/train.py --helpfull"
    command_string = 'python ' \
        + train_script \
        + ' --logtostderr' \
        + ' --train_split=%s '%train_split \
        + ' --base_learning_rate='+ str(base_learning_rate) \
        + ' --model_variant='+model_variant \
        + ' --atrous_rates='+str(atrous_rates1) \
        + ' --atrous_rates='+str(atrous_rates2) \
        + ' --atrous_rates='+str(atrous_rates3) \
        + ' --output_stride='+ str(output_stride) \
        + ' --decoder_output_stride=4 ' \
        + ' --train_crop_size=513' \
        + ' --train_crop_size=513' \
        + ' --train_batch_size='+str(batch_size) \
        + ' --training_number_of_steps=' + str(iteration_num)\
        + ' --fine_tune_batch_norm=False' \
        + ' --tf_initial_checkpoint=' +init_checkpoint \
        + ' --train_logdir='+train_logdir \
        + ' --dataset_dir='+dataset_dir \
        + ' --num_clones=' + str(gpu_num)

    return os.system(command_string)

# NUM_ITERATIONS=${iteration_num}
# python "${deeplab_dir}"/train.py \
#   --logtostderr \
#   --train_split="trainval" \
#   --base_learning_rate=${base_learning_rate} \
#   --model_variant="xception_65" \
#   --atrous_rates=${atrous_rates1} \
#   --atrous_rates=${atrous_rates2} \
#   --atrous_rates=${atrous_rates3} \
#   --output_stride=${output_stride} \
#   --decoder_output_stride=4 \
#   --train_crop_size=513 \
#   --train_crop_size=513 \
#   --train_batch_size=${batch_size} \
#   --training_number_of_steps="${NUM_ITERATIONS}" \
#   --fine_tune_batch_norm=False \
#   --tf_initial_checkpoint="${INIT_FOLDER}/xception/model.ckpt" \
#   --train_logdir="${TRAIN_LOGDIR}" \
#   --dataset_dir="${DATASET}" \
#   --num_clones=${gpu_num}

def evaluation_deeplab(evl_script,evl_split,model_variant,train_logdir, evl_logdir,dataset_dir, max_eva_number):

    # for information, run "python deeplab/eval.py  --helpfull"

# --eval_interval_secs: How often (in seconds) to run evaluation.
    #     (default: '300')
    #     (an integer)

    command_string = 'python ' \
                     + evl_script \
                     + ' --logtostderr' \
                     + ' --eval_split=%s ' % evl_split \
                     + ' --model_variant=' + model_variant \
                     + ' --atrous_rates=' + str(atrous_rates1) \
                     + ' --atrous_rates=' + str(atrous_rates2) \
                     + ' --atrous_rates=' + str(atrous_rates3) \
                     + ' --output_stride=' + str(output_stride) \
                     + ' --decoder_output_stride=4 ' \
                     + ' --eval_crop_size=513' \
                     + ' --eval_crop_size=513' \
                     + ' --checkpoint_dir=' + train_logdir \
                     + ' --eval_logdir=' + evl_logdir \
                     + ' --dataset_dir=' + dataset_dir \
                     + ' --max_number_of_evaluations=' + str(max_eva_number)

    return os.system(command_string)

        # python "${deeplab_dir}"/eval.py \
    # --logtostderr \
    # --eval_split="val" \
    # --model_variant="xception_65" \
    # --atrous_rates=${atrous_rates1} \
    # --atrous_rates=${atrous_rates2} \
    # --atrous_rates=${atrous_rates3} \
    # --output_stride=${output_stride} \
    # --decoder_output_stride=4 \
    # --eval_crop_size=513 \
    # --eval_crop_size=513 \
    # --checkpoint_dir="${TRAIN_LOGDIR}" \
    # --eval_logdir="${EVAL_LOGDIR}" \
    # --dataset_dir="${DATASET}" \
    # --max_number_of_evaluations=1

    pass

if __name__ == '__main__':
    print("%s : train deeplab" % os.path.basename(sys.argv[0]))

    para_file = sys.argv[1]
    gpu_num = int(sys.argv[2])
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
    sys.path.insert(0, code_dir)
    import parameters

    deeplabRS = parameters.get_directory_None_if_absence(para_file, 'deeplabRS_dir')
    sys.path.insert(0, deeplabRS)
    import basic_src.io_function as io_function


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

    # prepare training folder
    EXP_FOLDER = expr_name
    INIT_FOLDER =  os.path.join(WORK_DIR,EXP_FOLDER,'init_models')
    TRAIN_LOGDIR = os.path.join(WORK_DIR,EXP_FOLDER,'train')
    EVAL_LOGDIR = os.path.join(WORK_DIR,EXP_FOLDER,'eval')
    VIS_LOGDIR = os.path.join(WORK_DIR,EXP_FOLDER,'vis')
    EXPORT_DIR = os.path.join(WORK_DIR,EXP_FOLDER,'export')

    os.system('mkdir -p %s'%INIT_FOLDER)
    os.system('mkdir -p %s'%TRAIN_LOGDIR)
    os.system('mkdir -p %s'%EVAL_LOGDIR)
    os.system('mkdir -p %s'%VIS_LOGDIR)
    os.system('mkdir -p %s'%EXPORT_DIR)

    # prepare the tensorflow check point (pretrained model) for training
    pre_trained_dir = parameters.get_directory_None_if_absence(network_setting_ini, 'pre_trained_model_folder')
    pre_trained_tar = parameters.get_string_parameters(network_setting_ini, 'TF_INIT_CKPT')
    pre_trained_path = os.path.join(pre_trained_dir, pre_trained_tar)
    if os.path.isfile(pre_trained_path) is False:
        print('pre-trained model: %s not exist, try to download'%pre_trained_path)
        # try to download the file
        pre_trained_url = parameters.get_string_parameters_None_if_absence(network_setting_ini, 'pre_trained_model_url')
        os.system('wget %s '%pre_trained_url)
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


    # run training
    gpu_num = 1
    train_script = os.path.join(deeplab_dir, 'train.py')
    train_split = os.path.splitext(parameters.get_string_parameters(para_file,'training_sample_list_txt'))[0]
    model_variant = parameters.get_string_parameters(network_setting_ini, 'model_variant')
    checkpoint = parameters.get_string_parameters(network_setting_ini, 'tf_initial_checkpoint')
    init_checkpoint = os.path.join(INIT_FOLDER,checkpoint)
    train_deeplab(train_script, train_split, base_learning_rate, model_variant, init_checkpoint, TRAIN_LOGDIR,
                  dataset_dir, gpu_num,
                  atrous_rates1, atrous_rates2, atrous_rates3, output_stride, batch_size, iteration_num)


    # run evaluation









