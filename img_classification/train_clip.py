#!/usr/bin/env python
# Filename: clip_train.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 January, 2024
"""

import os,sys
from optparse import OptionParser
import time
from datetime import datetime
import torch
import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
import basic_src.timeTools as timeTools
import basic_src.basic as basic
from datetime import timedelta

from prediction_clip import prepare_dataset, run_prediction, calculate_top_k_accuracy
# from generate_pseudo_labels import generate_pseudo_labels
import class_utils

import clip
import torch.nn as nn
import torch.optim as optim

import logging
logger = logging.getLogger("Model")

def log_string(str):
    logger.info(str)
    print(str)


def evaluate(model, test_loader, device, prompt):
    """
    Evaluating the resulting classifier using a given test set loader.
    """

    pre_probs, gts = run_prediction(model, test_loader, prompt, device)

    top_probs_1, top_labels_1 = pre_probs.cpu().topk(1, dim=-1)
    top1_accuray = calculate_top_k_accuracy(top_labels_1, gts, k=1)

    top_probs_5, top_labels_5 = pre_probs.cpu().topk(5, dim=-1)
    top5_accuray = calculate_top_k_accuracy(top_labels_5, gts, k=5)

    return top1_accuray, top5_accuray


def prepare_training_data(WORK_DIR, para_file, transform, test=False):

    training_regions = parameters.get_string_list_parameters_None_if_absence(para_file,'training_regions')
    if training_regions is None or len(training_regions) < 1:
        raise ValueError('No training area is set in %s'%para_file)

    # TODO: support multiple training regions
    area_ini = training_regions[0]
    area_name = parameters.get_string_parameters(area_ini, 'area_name')
    area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
    area_time = parameters.get_string_parameters(area_ini, 'area_time')

    area_name_remark_time = area_name + '_' + area_remark + '_' + area_time
    area_save_dir = os.path.join(WORK_DIR, area_name_remark_time)

    # prepare training data
    train_image_dir = parameters.get_directory(area_ini, 'input_image_dir')
    train_image_or_pattern = parameters.get_string_parameters(area_ini, 'input_image_or_pattern')
    # TODO need to check preprocess, do we need to define it?
    in_dataset = prepare_dataset(para_file,area_ini,area_save_dir, train_image_dir, train_image_or_pattern,
                                 transform, test=test)
    return in_dataset

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def run_training_model(work_dir, network_ini, train_dataset, valid_dataset,prompt, device, model, preprocess, num_workers,description=''):

    # setting logger
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('%s/%s.txt' % (work_dir, 'train_log-%s-%s' % (description,timeTools.get_now_time_str())))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optionally resume training
    # if os.path.isfile(args.load_from):
    #     log_string("Loading pretrained model : [%s]" % args.load_from)
    #     checkpoint = torch.load(open(args.load_from, 'rb'), map_location="cpu")
    #     model.load_state_dict(checkpoint['state_dict'])


    if device == "cpu":
        model.float()

    batch_size = parameters.get_digit_parameters(network_ini,'batch_size','int')
    learning_rate =  parameters.get_digit_parameters(network_ini,'base_learning_rate','float')
    nbatches =  parameters.get_digit_parameters(network_ini,'train_epoch_num','int')

    # Defining Loss and Optimizer
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # need to read this from network_ini file
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6,
                           weight_decay=0.2)  # the lr is smaller, more safe for fine tuning to new dataset
    decay_step = 20     # decay-step to use
    decay = 0.7         # decay rate to use
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_step, gamma=decay)

    # Instantiating data loaders
    # train_transforms = T.Compose([
    #     T.RandomHorizontalFlip(),
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize((0.48422758, 0.49005175, 0.45050276), (0.17348297, 0.16352356, 0.15547496)), #recompute
    # ])

    # train_transforms = T.Compose([
    #     T.Resize(256),
    #     T.CenterCrop(224),
    #     T.RandomHorizontalFlip(),
    #     # T.ColorJitter(0.05, 0.05, 0.05),
    #     # T.RandomRotation(10),
    #     T.ToTensor(),
    #     T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    # ])
    #
    # test_transforms = T.Compose([
    #     T.Resize(256),
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    # ])



    log_string('train size: {}/test size: {}'.format(len(train_dataset), len(valid_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    # Training
    starting_time = time.time()

    # Main loop
    tstart = time.time()

    log_string("Starting training...")
    log_string("Number of batches: [%d]" % (len(train_loader.dataset) / batch_size))

    n_batch = 0
    loss = None
    saved_model = None

    while n_batch <= nbatches:
        # Training the model
        total_top1_hits, total_top5_hits, N = 0, 0, 0
        top1_avg, top5_avg = 0, 0

        for images, targets, _ in train_loader:

            ## Making a checkpoint
            if n_batch % 100 == 0:

                # Measuring model test-accuracy
                top1_test_acc, top5_test_acc = evaluate(model, test_loader, device, prompt)
                log_string('Evaluation {:03}/{:03}, top1_test_acc: {:.3f}, top5_test_acc: {:.3f}'.
                           format(n_batch,nbatches,top1_test_acc,top5_test_acc))

                if n_batch == 300:
                    saved_model = os.path.join(work_dir, "batch_%d_%s.ckpt" % (n_batch,description))
                    log_string("Saved model at: [" + saved_model + "]")
                    state = {
                        "top1_train_acc": top1_avg,
                        "top5_train_acc": top5_avg,
                        "top1_test_acc": top1_test_acc,
                        "top5_test_acc": top5_test_acc,
                        "state_dict": model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, saved_model)

            # pdb.set_trace()
            model.train()
            optimizer.zero_grad()

            images, targets = images.to(device), targets.to(device)
            texts = [prompt.format(train_loader.dataset.classes[t]) for t in targets]
            texts = torch.stack([clip.tokenize(t) for t in texts])
            texts = texts.squeeze().to(device)

            ## Forward + backward + optimize
            logits_per_image, logits_per_text = model(images, texts)  # logits_per_image_orig
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            loss.backward()

            if device == torch.device("cpu"):
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            ## Logging results
            outputs = logits_per_image.softmax(dim=-1)
            top5_hits, top1_hits = class_utils.calculate_metrics(outputs, ground_truth)
            total_top1_hits += top1_hits
            total_top5_hits += top5_hits
            N += images.shape[0]
            top1_avg = 100 * (float(total_top1_hits) / N)
            top5_avg = 100 * (float(total_top5_hits) / N)

            log_string('Training {:03}/{:03} | loss = {:.4f} | top-1 acc = {:.3f} | top-5 acc = {:.3f}'.
                       format(n_batch,nbatches,loss.item(),top1_avg,top5_avg))

            n_batch += 1
            scheduler.step()


            if (n_batch >= nbatches): break

        # Logging results
        current_elapsed_time = time.time() - starting_time
        log_string('{:03}/{:03} | {} | Train : loss = {:.4f} | top-1 acc = {:.3f} | top-5 acc = {:.3f}'.
                   format(n_batch, nbatches,
                          timedelta(seconds=round(current_elapsed_time)),
                          loss, top1_avg, top5_avg))
        # clean (try to avoid memory issues)
        del loss, outputs
        del top1_avg, top5_avg,total_top1_hits, total_top5_hits, N

    # Final output
    log_string('[Elapsed time = {:.1f} min]'.format((time.time() - tstart) / 60))
    log_string('Done!')
    logger.removeHandler(file_handler)
    file_handler.close()
    # clear unused memory
    torch.cuda.empty_cache()

    return saved_model



def training_zero_shot(para_file, network_ini, WORK_DIR, train_save_dir, device, model, preprocess):
    # without any human input training data
    dataset = prepare_training_data(WORK_DIR, para_file, preprocess, test=True)

    num_workers = parameters.get_digit_parameters(para_file,'process_num','int')
    train_batch_size = parameters.get_digit_parameters(network_ini,'batch_size','int')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)


    # probs_thr=0.6, topk=10, version=1
    topk_list = parameters.get_string_list_parameters_None_if_absence(network_ini,'topk_list')
    probability_threshold = parameters.get_digit_parameters(network_ini,'probability_threshold','float')
    if topk_list is not None:
        topk_list = [ int(item) for item in topk_list ]
    else:
        topk = parameters.get_digit_parameters(network_ini,'topk','int')
        topk_list = [topk]

    previous_train_model = None
    # TODO: it doesn't release CUDA memory at each iteration, eventually, out of CUDA memory.
    for v_num, topk in enumerate(topk_list, start=1):
        # get pseudo labels
        clip_prompt = parameters.get_string_parameters(para_file, 'clip_prompt')
        training_samples_txt = generate_pseudo_labels(dataset, data_loader, train_save_dir, device, model,clip_prompt,
                                                      probs_thr=probability_threshold, topk=topk,version=v_num)

        # prepare new training datasets using pseudo labels
        class_labels = parameters.get_file_path_parameters(para_file, 'class_labels')
        image_path_labels = [item.split() for item in io_function.read_list_from_txt(training_samples_txt)]
        image_path_list = [item[0] for item in image_path_labels]   # it's already absolute path
        image_labels = [int(item[1]) for item in image_path_labels]
        train_dataset = class_utils.RSPatchDataset(image_path_list, image_labels, label_txt=class_labels, transform=preprocess, test=True)

        # load models from previous iteration?
        if previous_train_model is not None:
            log_string("Loading pretrained model : [%s]" % previous_train_model)
            checkpoint = torch.load(open(previous_train_model, 'rb'), map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'])

        # run training
        description = 'v{}_topk{}'.format(v_num,topk)
        save_model = run_training_model(train_save_dir, network_ini, train_dataset, dataset,clip_prompt, device, model, preprocess, num_workers,
                                        description = description)
        previous_train_model = save_model

        # clear memory
        torch.cuda.empty_cache()


def training_zero_shot_bash(para_file, network_ini, WORK_DIR, train_save_dir, device, model, preprocess):
    # without any human input training data
    # similar to training_zero_shot, but use to run the training, avoid out-of-memory problem

    # probs_thr=0.6, topk=10, version=1
    topk_list = parameters.get_string_list_parameters_None_if_absence(network_ini, 'topk_list')
    # probability_threshold = parameters.get_digit_parameters(network_ini, 'probability_threshold', 'float')
    if topk_list is not None:
        topk_list = [int(item) for item in topk_list]
    else:
        topk = parameters.get_digit_parameters(network_ini, 'topk', 'int')
        topk_list = [topk]

    previous_train_model = None
    pydir = os.path.dirname(os.path.abspath(__file__))
    py_train = os.path.abspath(__file__)
    py_get_pseudo = os.path.join(pydir, 'generate_pseudo_labels.py')


    for v_num, topk in enumerate(topk_list, start=1):
        # get pseudo labels
        cmd_str = py_get_pseudo + ' ' + para_file + ' ' + str(v_num) + ' ' + str(topk)
        if previous_train_model is not None:
            cmd_str += ' --trained_model=%s '%previous_train_model
        basic.os_system_exit_code(cmd_str)


        save_path_txt = class_utils.get_pseudo_labels_path(train_save_dir,v_num, topk)
        if os.path.isfile(save_path_txt) is False:
            raise ValueError('generating %s failed'%save_path_txt)

        # run training
        cmd_str = py_train + ' ' + para_file + ' ' + ' --train_data_txt=%s '%save_path_txt + ' --b_a_few_shot '
        if previous_train_model is not None:
            cmd_str += ' --pretrain_model=%s '%previous_train_model
        basic.os_system_exit_code(cmd_str)

        save_model_path = class_utils.get_model_save_path(train_save_dir,para_file,save_path_txt)
        if os.path.isfile(save_model_path) is False:
            raise ValueError('find trained model %s failed' % save_model_path)
        previous_train_model = save_model_path

        # clear memory
        torch.cuda.empty_cache()


def training_few_shot(para_file, network_ini, WORK_DIR, train_save_dir, device, model, preprocess,p_train_model='', train_data_txt=''):
    # with a few human input training data
    dataset = prepare_training_data(WORK_DIR, para_file, preprocess, test=False)

    num_workers = parameters.get_digit_parameters(para_file, 'process_num', 'int')
    train_batch_size = parameters.get_digit_parameters(network_ini, 'batch_size', 'int')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    # get pseudo labels
    clip_prompt = parameters.get_string_parameters(para_file, 'clip_prompt')

    if os.path.isfile(train_data_txt):
        # prepare new training datasets
        class_labels = parameters.get_file_path_parameters(para_file, 'class_labels')
        image_path_labels = [item.split() for item in io_function.read_list_from_txt(train_data_txt)]
        image_path_list = [item[0] for item in image_path_labels]  # it's already absolute path
        image_labels = [int(item[1]) for item in image_path_labels]
        train_dataset = class_utils.RSPatchDataset(image_path_list, image_labels, label_txt=class_labels,
                                                   transform=preprocess, test=False)
    else:
        # TODO: split dataset into training and validation
        train_dataset = dataset

    # resume training, need to read the trained model from the disks
    if os.path.isfile(p_train_model):
        log_string("Loading pretrained model : [%s]" % p_train_model)
        checkpoint = torch.load(open(p_train_model, 'rb'), map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])

    # run training
    description = 'few_shot'
    save_model = run_training_model(train_save_dir, network_ini, train_dataset, train_dataset, clip_prompt, device, model,
                                    preprocess, num_workers,
                                    description=description)
    torch.cuda.empty_cache()

    # rename save file path
    save_model_path = class_utils.get_model_save_path(train_save_dir, para_file, train_data_txt)
    io_function.move_file_to_dst(save_model, save_model_path)


def train_clip(WORK_DIR, para_file,pre_train_model='',train_data_txt='',b_a_few_shot=False,gpu_num=1):
    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = parameters.get_string_parameters(network_ini, 'model_type')
    model, preprocess = clip.load(model_type, device=device)

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    if b_a_few_shot:
        b_a_few_shot_training = True
    else:
        b_a_few_shot_training = parameters.get_bool_parameters(para_file, 'a_few_shot_training')

    if b_a_few_shot_training:
        training_few_shot(para_file, network_ini, WORK_DIR, train_save_dir, device, model, preprocess,
                          p_train_model=pre_train_model,train_data_txt=train_data_txt)
    else:
        # training_zero_shot(para_file, network_ini, WORK_DIR, train_save_dir, device, model, preprocess )
        training_zero_shot_bash(para_file, network_ini, WORK_DIR, train_save_dir, device, model, preprocess)

    # result backup

def clip_train_main(para_file,pre_train_model='',train_data_txt='',b_a_few_shot=False,gpu_num=1):
    print(datetime.now(),"train CLIP")
    SECONDS = time.time()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    WORK_DIR = os.getcwd()
    train_clip(WORK_DIR, para_file, pre_train_model=pre_train_model,train_data_txt=train_data_txt,
                    b_a_few_shot=b_a_few_shot,gpu_num=gpu_num)

    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of training: %.2f seconds">>time_cost.txt' % duration)


def main(options, args):

    para_file = args[0]
    pre_train_model = options.pretrain_model
    train_data_txt = options.train_data_txt
    b_a_few_shot = options.b_a_few_shot

    clip_train_main(para_file,pre_train_model=pre_train_model,train_data_txt=train_data_txt,
                    b_a_few_shot=b_a_few_shot)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-01-24")
    parser.description = 'Introduction: fine-tune the clip model using custom data'

    parser.add_option("-m", "--pretrain_model",
                      action="store", dest="pretrain_model",default='',
                      help="the pre-trained model")

    parser.add_option("-t", "--train_data_txt",
                      action="store", dest="train_data_txt",default='',
                      help="the training dataset saved in txt")

    parser.add_option("-f", "--b_a_few_shot",
                      action="store_true", dest="b_a_few_shot", default=False,
                      help="if set, will force to run a few shot training, ignoring the the setting in ini files")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)