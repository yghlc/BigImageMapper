#!/usr/bin/env python
# Filename: zeroshort_classify_clip.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 January, 2024
"""



import os,sys
import os.path as osp
from optparse import OptionParser
from datetime import datetime
import time
import GPUtil

from PIL import Image

import numpy as np
import torch
import clip

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic
import basic_src.timeTools as timeTools

from multiprocessing import Process
# import torch.multiprocessing as Process

from class_utils import RSPatchDataset
import class_utils

from tqdm import tqdm

def is_file_exist_in_folder(folder):
    # just check if the folder is empty
    if len(os.listdir(folder)) == 0:
        return False
    else:
        return True

def calculate_top_k_accuracy(predict_labels,ground_truths, save_path=None, k=5):
    if torch.is_tensor(ground_truths):
        ground_truths = ground_truths.cpu().numpy()
    if torch.is_tensor(predict_labels):
        predict_labels = predict_labels.cpu().numpy()

    topk_accuray = 0.0

    if np.all(ground_truths == -1):
        print_msg = 'No ground truth, skip reporting accuracy for %d prediction'%len(predict_labels)
    elif np.any(ground_truths == -1):
        print_msg = 'Some ground truth is missing, skip reporting accuracy for %d prediction'%len(predict_labels)
    else:
        # top-k accuracy
        if k > 1:
            predict_labels = predict_labels.squeeze()
        # print(top_labels_5)
        hit_count = 0
        for gt, pred_l_s in zip(ground_truths,predict_labels):
            # print(pred_l_s)
            if gt in pred_l_s:
                hit_count += 1
        topk_accuray = 100.0*hit_count/len(ground_truths)
        print_msg = 'top %d accuracy: (%d /%d): %f'%(k, hit_count, len(ground_truths), topk_accuray)

    print(print_msg)
    if save_path is not None:
        io_function.save_list_to_txt(save_path,[print_msg])
    return topk_accuray


def save_prediction_results(dataset, predict_probs, save_path, k=5):
    if k < 1:
        raise ValueError('k should be larger than 0')
    top_probs_k, top_labels_k = predict_probs.cpu().topk(5, dim=-1)
    top_probs_k = top_probs_k.numpy().squeeze()
    top_labels_k = top_labels_k.numpy().squeeze()

    # save to a json file
    res_dict = {}
    for i_path, probs, labels in zip(dataset.img_list,top_probs_k,top_labels_k):
        res_dict[os.path.basename(i_path)] = { }
        res_dict[os.path.basename(i_path)]['confidence'] = probs.tolist()
        res_dict[os.path.basename(i_path)]['pre_labels'] = labels.tolist()

    io_function.save_dict_to_txt_json(save_path,res_dict)


def test_classification_ucm(model, preprocess):
    data_dir = os.path.expanduser('~/Data/image_classification/UCMerced_LandUse')


    # read classes info
    label_list_txt = os.path.join(data_dir,'label_list.txt')
    class_labels = [item.split(',')[0] for item in io_function.read_list_from_txt(label_list_txt) ]
    text_descriptions = [f"This is a satellite image of a {label}" for label in class_labels]
    text_tokens = clip.tokenize(text_descriptions).cuda()

    # process text
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)


    # randomly read ten images
    image_txt = os.path.join(data_dir,'all.txt')
    image_list = [ item.split() for item in io_function.read_list_from_txt(image_txt)]
    image_path_list = [ os.path.join(data_dir,'Images', item[0]) for item in image_list]
    image_class_list = [ int(item[1]) for item in image_list]

    images = []
    #sel_index = [0, 10, 100, 200, 300, 500, 700, 900, 1000,1500, 2000]
    sel_index = [item for item in range(len(image_path_list))]
    for idx in sel_index:
        image = Image.open(image_path_list[idx]).convert("RGB")
        images.append(preprocess(image))
    image_input = torch.tensor(np.stack(images)).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs_5, top_labels_5 = text_probs.cpu().topk(5, dim=-1)
    print(top_probs_5)
    print(top_labels_5)

    top_probs_1, top_labels_1 = text_probs.cpu().topk(1, dim=-1)
    print(top_probs_1)
    print(top_labels_1)

    # output accuracy
    # top1 accuracy
    ground_truths = [image_class_list[idx] for idx in sel_index]
    calculate_top_k_accuracy(top_labels_1, ground_truths, k=1)

    # top5 accuracy
    calculate_top_k_accuracy(top_labels_5, ground_truths, k=5)

def prepare_dataset(para_file, area_ini, area_save_dir, image_dir, image_or_pattern, transform=None, test = False,
                    extract_img_dir=None, training_poly_shp=None):
    area_data_type = parameters.get_string_parameters(area_ini,'area_data_type')
    inf_image_dir = image_dir
    inf_image_or_pattern = image_or_pattern
    class_labels = parameters.get_file_path_parameters(para_file,'class_labels')
    if extract_img_dir is None:
        extract_img_dir = os.path.join(os.getcwd(),'image_patches', os.path.basename(area_save_dir))

    if area_data_type == 'image_patch':
        all_image_patch_labels = parameters.get_file_path_parameters(area_ini, 'all_image_patch_labels')
        image_path_labels = [item.split() for item in io_function.read_list_from_txt(all_image_patch_labels)]
        # image_path_labels = image_path_labels[:200] # for test
        if 'UCMerced_LandUse' in inf_image_dir:
            inf_image_dir = os.path.join(inf_image_dir, 'Images')
        image_path_list = [os.path.join(inf_image_dir, item[0]) for item in image_path_labels]
        image_labels = [ int(item[1]) for item in image_path_labels]

        input_data = RSPatchDataset(image_path_list, image_labels, label_txt=class_labels, transform=transform, test = test)

    elif area_data_type == 'image_vector':

        # extract sub-images
        get_subImage_script = os.path.join(code_dir, 'datasets', 'get_subImages.py')

        dstnodata = parameters.get_string_parameters(para_file, 'dst_nodata')
        buffersize = parameters.get_string_parameters(para_file, 'buffer_size')
        rectangle_ext = parameters.get_string_parameters(para_file, 'b_use_rectangle')
        process_num = parameters.get_digit_parameters(para_file, 'process_num', 'int')

        extract_done_indicator = os.path.join(extract_img_dir,'extract_image_using_vector.done')
        patch_list_txt = os.path.join(extract_img_dir, os.path.basename(area_save_dir) + '_patch_list.txt')

        if training_poly_shp is not None:
            # assign a different training polygon
            all_polygons_labels = training_poly_shp
        else:
            all_polygons_labels = parameters.get_file_path_parameters_None_if_absence(area_ini,'all_polygons_labels')
        if all_polygons_labels is not None:
            command_string = get_subImage_script  + ' -b ' + str(buffersize) + ' -e ' + inf_image_or_pattern + \
                             ' -o ' + extract_img_dir + ' -n ' + str(dstnodata) + ' -p ' + str(process_num) \
                             + ' ' + rectangle_ext + ' --no_label_image ' + all_polygons_labels + ' ' + inf_image_dir
            if os.path.isfile(extract_done_indicator):
                basic.outputlogMessage('Warning, sub-images already been extracted, read them directly')
            else:
                basic.os_system_exit_code(command_string)
            image_path_list = io_function.get_file_list_by_pattern(extract_img_dir, 'subImages/*.tif')
            image_labels = class_utils.get_class_labels_from_vector_file(image_path_list, all_polygons_labels)
        else:
            # get sub-images, grid by grid
            all_polygons_dir = parameters.get_directory(area_ini,'all_polygons_dir')
            all_polygons_pattern = parameters.get_string_parameters(area_ini,'all_polygons_pattern')
            vector_file_list = class_utils.get_file_list(all_polygons_dir,all_polygons_pattern,area_ini)
            raster_file_list = class_utils.get_file_list(inf_image_dir,inf_image_or_pattern,area_ini)

            image_path_list = []
            image_labels = []

            # pair the vector file and raster files
            raster_vector_pairs = class_utils.pair_raster_vecor_files_grid(vector_file_list, raster_file_list)
            for key in raster_vector_pairs:
                vector_file = raster_vector_pairs[key][0]
                raster_file = raster_vector_pairs[key][1]
                grid_save_dir = os.path.join(extract_img_dir, 'grid%d'%key)
                command_string = get_subImage_script + ' -b ' + str(buffersize) + ' -e ' + os.path.basename(raster_file) + \
                                 ' -o ' + grid_save_dir + ' -n ' + str(dstnodata) + ' -p ' + str(process_num) \
                                 + ' ' + rectangle_ext + ' --no_label_image ' + vector_file + ' ' + os.path.dirname(raster_file)
                if os.path.isfile(extract_done_indicator):
                    basic.outputlogMessage('Warning, sub-images already been extracted, read them directly')
                else:
                    basic.os_system_exit_code(command_string)

                image_path_list_grid = io_function.get_file_list_by_pattern(grid_save_dir, 'subImages/*.tif')
                image_labels_grid = class_utils.get_class_labels_from_vector_file(image_path_list_grid, vector_file)

                image_path_list.extend(image_path_list_grid)
                image_labels.extend(image_labels_grid)

        if os.path.isfile(patch_list_txt) is False:
            # save the relative path and label to file
            image_path_label_list = [ '%s %d'%(os.path.realpath(item), idx) for idx, item in zip(image_labels, image_path_list) ]
            io_function.save_list_to_txt(patch_list_txt ,image_path_label_list)


        if os.path.isfile(extract_done_indicator) is False:
            with open(extract_done_indicator,'w') as f_obj:
                f_obj.writelines('%s image extracting, complete on %s \n'% (extract_img_dir, timeTools.get_now_time_str() ))

        input_data = RSPatchDataset(image_path_list, image_labels, label_txt=class_labels, transform=transform, test = test)
    else:
        raise ValueError('Unknown area data type: %s, only accept: image_patch and image_vector'%area_data_type)

    basic.outputlogMessage('read %d images for prediction'%len(input_data))
    return input_data

def run_prediction(model, test_loader,prompt, device):

    model.eval()
    model.float()
    text_descriptions = [prompt.format(label) for label in test_loader.dataset.classes]
    # text_tokens = clip.tokenize(text_descriptions).cuda()
    text_tokens = clip.tokenize(text_descriptions).to(device)

    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    pre_probs = []
    gts = []
    with torch.no_grad():
        cnt = 0
        for data in tqdm(test_loader):
            images, targets, _ = data
            images = images.to(device)
            targets = targets.to(device).squeeze()

            image_features = model.encode_image(images)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pre_probs.append(similarity)    # .cpu().numpy()
            gts.append(targets)             #.cpu().numpy()


    # pre_probs = np.concatenate(pre_probs, 0)     # for numpy array
    pre_probs = torch.cat(pre_probs, 0)               # for tensor
    gts = torch.cat(gts, 0)               # for tensor
    return pre_probs, gts


def predict_remoteSensing_data(para_file, area_idx, area_ini, area_save_dir,model_type, trained_model, batch_size=16):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_type,device=device)

    # load trained model
    if os.path.isfile(trained_model):
        checkpoint = torch.load(open(trained_model, 'rb'), map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])

    if device == "cpu":
        model.eval()
    else:
        model.cuda().eval()  # to download the pre-train models.

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    # run image classification
    inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')
    inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')
    in_dataset = prepare_dataset(para_file, area_ini,area_save_dir,inf_image_dir, inf_image_or_pattern,
                                 transform=preprocess,test=True)
    clip_prompt = parameters.get_string_parameters(para_file,'clip_prompt')

    # TODO: read num_workers from para_file
    test_loader = torch.utils.data.DataLoader(
        in_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    pre_probs, ground_truths = run_prediction(model, test_loader, clip_prompt, device)

    save_path = os.path.join(area_save_dir, os.path.basename(area_save_dir)+'-classify_results.json' )
    save_prediction_results(in_dataset,pre_probs, save_path, k=5)

    top_probs_5, top_labels_5 = pre_probs.cpu().topk(5, dim=-1)
    # print(top_probs_5)
    # print(top_labels_5)

    top_probs_1, top_labels_1 = pre_probs.cpu().topk(1, dim=-1)
    # print(top_probs_1)
    # print(top_labels_1)

    # output accuracy
    # top1 accuracy
    top1_acc_save_path = os.path.join(area_save_dir, 'top1_accuracy.txt' )
    calculate_top_k_accuracy(top_labels_1, ground_truths, save_path=top1_acc_save_path, k=1)

    # top5 accuracy
    top5_acc_save_path = os.path.join(area_save_dir, 'top5_accuracy.txt')
    calculate_top_k_accuracy(top_labels_5, ground_truths, save_path=top5_acc_save_path, k=5)






def classify_one_region(area_idx, area_ini, para_file, area_save_dir, gpuid, inf_list_file, model_type, trained_model):

    inf_batch_size = parameters.get_digit_parameters(para_file,'inf_batch_size','int')

    done_indicator = '%s_done'%inf_list_file
    if os.path.isfile(done_indicator):
        basic.outputlogMessage('warning, %s exist, skip prediction'%done_indicator)
        return
    # use a specific GPU for prediction, only inference one image
    time0 = time.time()
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    predict_remoteSensing_data(para_file, area_idx, area_ini, area_save_dir,model_type, trained_model, batch_size=inf_batch_size)

    duration = time.time() - time0
    os.system('echo "$(date): time cost of inference for image in %s: %.2f seconds">>"time_cost.txt"' % (inf_list_file, duration))
    # write a file to indicate that the prediction has done.
    os.system('echo %s > %s_done'%(inf_list_file,inf_list_file))

def parallel_prediction_main(para_file,trained_model):

    print("CLIP prediction using the trained model ") # (run parallel if using multiple GPUs)
    machine_name = os.uname()[1]
    start_time = datetime.now()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    model_type = parameters.get_string_parameters(network_ini,'model_type')

    outdir = os.path.join(parameters.get_directory(para_file, 'inf_output_dir'), expr_name)
    # remove previous results (let user remove this folder manually or in exe.sh folder)
    io_function.mkdir(outdir)

    # get name of inference areas
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')
    b_use_multiGPUs = parameters.get_bool_parameters(para_file, 'b_use_multiGPUs')
    maximum_prediction_jobs = parameters.get_digit_parameters(para_file, 'maximum_prediction_jobs', 'int')

    # loop each inference regions
    sub_tasks = []
    for area_idx, area_ini in enumerate(multi_inf_regions):

        area_name = parameters.get_string_parameters(area_ini, 'area_name')
        area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
        area_time = parameters.get_string_parameters(area_ini, 'area_time')

        inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')

        # it is ok consider a file name as pattern and pass it the following functions to get file list
        inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')

        inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir, inf_image_or_pattern)
        img_count = len(inf_img_list)
        if img_count < 1:
            raise ValueError(
                'No image for inference, please check inf_image_dir (%s) and inf_image_or_pattern (%s) in %s'
                % (inf_image_dir, inf_image_or_pattern, area_ini))

        area_name_remark_time = area_name + '_' + area_remark + '_' + area_time
        area_save_dir = os.path.join(outdir, area_name_remark_time)
        io_function.mkdir(area_save_dir)

        # parallel inference images for this area
        CUDA_VISIBLE_DEVICES = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            CUDA_VISIBLE_DEVICES = [int(item.strip()) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        # idx = 0

        while basic.alive_process_count(sub_tasks) >= maximum_prediction_jobs:
            print(datetime.now(),
                  '%d jobs are running simultaneously, wait 5 seconds' % basic.alive_process_count(sub_tasks))
            time.sleep(5)  # wait 5 seconds, then check the count of running jobs again

        if b_use_multiGPUs:
            # get available GPUs  # https://github.com/anderskm/gputil
            # memory: orders the available GPU device ids by ascending memory usage
            deviceIDs = GPUtil.getAvailable(order='memory', limit=100, maxLoad=0.5,
                                            maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
            # only use the one in CUDA_VISIBLE_DEVICES
            if len(CUDA_VISIBLE_DEVICES) > 0:
                deviceIDs = [item for item in deviceIDs if item in CUDA_VISIBLE_DEVICES]
                basic.outputlogMessage('on ' + machine_name + ', available GPUs:' + str(deviceIDs) +
                                       ', among visible ones:' + str(CUDA_VISIBLE_DEVICES))
            else:
                basic.outputlogMessage('on ' + machine_name + ', available GPUs:' + str(deviceIDs))

            if len(deviceIDs) < 1:
                time.sleep(5)  # wait 5 seconds, then check the available GPUs again
                continue
            # set only the first available visible
            gpuid = deviceIDs[0]
            basic.outputlogMessage(
                '%d: predict region: %s on GPU %d of %s' % (area_idx, area_name_remark_time, gpuid, machine_name))
        else:
            gpuid = None
            basic.outputlogMessage('%d: predict region: %s on %s' % (area_idx, area_name_remark_time, machine_name))

        # run inference
        inf_list_file = os.path.join(area_save_dir, '%d.txt' % area_idx)

        done_indicator = '%s_done' % inf_list_file
        if os.path.isfile(done_indicator):
            basic.outputlogMessage('warning, %s exist, skip prediction' % done_indicator)
            continue

        # if it already exists, then skip
        if os.path.isdir(area_save_dir) and is_file_exist_in_folder(area_save_dir):
            basic.outputlogMessage('folder of %dth region (%s) already exist, '
                                   'it has been predicted or is being predicted' % (area_idx, area_name_remark_time))
            continue

        with open(inf_list_file, 'w') as inf_obj:
            inf_obj.writelines(area_name_remark_time + '\n')

        sub_process = Process(target=classify_one_region,
                              args=(area_idx, area_ini, para_file, area_save_dir, gpuid, inf_list_file, model_type, trained_model))

        sub_process.start()
        sub_tasks.append(sub_process)

        if b_use_multiGPUs is False:
            # wait until previous one finished
            while sub_process.is_alive():
                time.sleep(1)

        # wait until predicted image patches exist or exceed 20 minutes
        time0 = time.time()
        elapsed_time = time.time() - time0
        while elapsed_time < 20 * 60:
            elapsed_time = time.time() - time0
            file_exist = os.path.isdir(area_save_dir) and is_file_exist_in_folder(area_save_dir)
            if file_exist is True or sub_process.is_alive() is False:
                break
            else:
                time.sleep(1)

        if sub_process.exitcode is not None and sub_process.exitcode != 0:
            sys.exit(1)

        basic.close_remove_completed_process(sub_tasks)
            # if 'chpc' in machine_name:
            #     time.sleep(60)  # wait 60 second on ITSC services
            # else:
            #     time.sleep(10)

        # copy and backup area ini file
        bak_area_ini = os.path.join(area_save_dir, os.path.basename(io_function.get_name_by_adding_tail(area_ini, 'region')))
        io_function.copy_file_to_dst(area_ini, bak_area_ini, overwrite=True)


    # check all the tasks already finished
    wait_all_finish = 0
    while basic.b_all_process_finish(sub_tasks) is False:
        if wait_all_finish % 100 == 0:
            basic.outputlogMessage('wait all tasks to finish')
        time.sleep(1)
        wait_all_finish += 1

    basic.close_remove_completed_process(sub_tasks)
    end_time = datetime.now()

    diff_time = end_time - start_time
    out_str = "%s: time cost of total parallel inference on %s: %d seconds" % (
        str(end_time), machine_name, diff_time.total_seconds())
    basic.outputlogMessage(out_str)
    with open("time_cost.txt", 'a') as t_obj:
        t_obj.writelines(out_str + '\n')

    # copy and back up parameter files
    WORK_DIR = os.getcwd()
    test_id = os.path.basename(WORK_DIR) + '_' + expr_name
    bak_para_ini = os.path.join(outdir, '_'.join([test_id, 'para']) + '.ini')
    bak_network_ini = os.path.join(outdir, '_'.join([test_id, 'network']) + '.ini')
    bak_time_cost = os.path.join(outdir, '_'.join([test_id, 'time_cost']) + '.txt')

    io_function.copy_file_to_dst(para_file, bak_para_ini,overwrite=True)
    io_function.copy_file_to_dst(network_ini, bak_network_ini,overwrite=True)
    if os.path.isfile('time_cost.txt'):
        io_function.copy_file_to_dst('time_cost.txt', bak_time_cost,overwrite=True)



def main(options, args):

    para_file = args[0]
    trained_model = options.trained_model

    parallel_prediction_main(para_file, trained_model)


    # test_classification_ucm(model, preprocess)

if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-01-22")
    parser.description = 'Introduction: run prediction in parallel using clip '

    parser.add_option("-m", "--trained_model",
                      action="store", dest="trained_model", default='',
                      help="the trained model for prediction")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    torch.multiprocessing.set_start_method('spawn')

    main(options, args)
