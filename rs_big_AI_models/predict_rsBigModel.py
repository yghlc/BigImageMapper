#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Introduction:

Author: Huang Lingcao
Email: huanglingcao@gmail.com
Created: 2026-03-23
"""


import gc
from pyexpat import model
import os,sys
from optparse import OptionParser

import numpy as np
import torch

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

# for files in img_classification
code_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img_classification')
sys.path.insert(0, code_dir2)
# print(f"code_dir: {code_dir}, code_dir2: {code_dir2}")

import parameters
import bim_utils
import basic_src.basic as basic
import basic_src.io_function as io_function
import basic_src.timeTools as timeTools
from rsBigModel_utils import get_data_transforms, prepare_dataset_for_classify, RSPatchTxtModule
from img_classification.prediction_class import save_prediction_results, calculate_top_k_accuracy, read_label_ids_local, select_sample_for_manu_check

from datetime import datetime
import time

from multiprocessing import Process


from torch.utils.data import DataLoader, Dataset
from albumentations import Resize
from albumentations.pytorch.transforms import ToTensorV2

# Constants
IMAGE_PATH = os.path.expanduser("~/Data/public_data_AI/Landslide4Sense/TestData/img/image_174.h5")
CKPT_PATH = "./map_landslide/checkpoints/best-checkpoint-epoch=09-val_loss=0.00.ckpt"  # Replace with your checkpoint path
BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]


# Custom Dataset for Prediction
class LandslidePredictionDataset(Dataset):
    def __init__(self, image_path, bands):
        self.image_path = image_path
        self.bands = bands
        self.transform = Resize(224, 224)  # Resize to match model input size

    def __len__(self):
        return 1  # Single image

    def __getitem__(self, idx):
        # Load the image
        with h5py.File(self.image_path, "r") as f:
            img = f["img"][:]

        print(f"Loaded image shape: {img.shape}")

        # Extract the required bands
        band_indices = [BANDS.index(band) for band in self.bands]
        img_bands = img[ :, :, band_indices]

        # Normalize the image (assuming pixel values are in [0, 255])
        img_bands = img_bands / 255.0
        print(f"Image shape before resizing: {img_bands.shape}")

        # Apply resizing transformation
        transformed = self.transform(image=img_bands) # .transpose(2, 0, 1)
        print(f"Image shape after transformation: {transformed['image'].shape}")
        img_resized = transformed["image"]
        print(f"Image shape after resizing: {img_resized.shape}")

        # Convert to tensor and return
        to_tensor = ToTensorV2()
        img_tensor = to_tensor(image=img_resized)["image"]

        return img_tensor


# Visualize Results
def visualize_results(original_image, prediction_mask):
    plt.figure(figsize=(10, 5))

    # Original image (select RGB channels for visualization)
    plt.subplot(1, 2, 1)
    plt.title("Original Image (RGB)")
    rgb_img = original_image[[BANDS.index("RED"), BANDS.index("GREEN"), BANDS.index("BLUE")], :, :].transpose(1, 2, 0)
    plt.imshow(rgb_img)
    plt.axis("off")

    # Prediction mask
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(prediction_mask, cmap="jet")
    plt.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig("prediction_results.png")  # Save the figure if needed



def test_SemanticSegmentation():

    from terratorch.datamodules import Landslide4SenseNonGeoDataModule
    from terratorch.datasets import Landslide4SenseNonGeo
    from terratorch.tasks import SemanticSegmentationTask

    import lightning.pytorch as pl

     # Load the trained model
    # import terratorch
    # # from terratorch.tasks import SemanticSegmentationTask
    # from terratorch.datamodules import Landslide4SenseNonGeoDataModule

    # model= SemanticSegmentationTask().load_
    # from fine_tune_PrithviEO import get_deeplearning_model
    # model = get_deeplearning_model(BANDS)
    # model.load_from_checkpoint(CKPT_PATH)
    model = SemanticSegmentationTask.load_from_checkpoint(CKPT_PATH)

    # model = model.load_from_checkpoint(CKPT_PATH)
    DATASET_PATH = "/home/hlc/Data/public_data_AI/Landslide4Sense/data"
    data_module = Landslide4SenseNonGeoDataModule(data_root=DATASET_PATH, bands=BANDS, num_workers=8)
    data_module.setup(stage='predict')

    # # Create the dataset and dataloader
    # dataset = LandslidePredictionDataset(IMAGE_PATH, BANDS)
    # dataloader = DataLoader(dataset, batch_size=1)

    # Initialize the Trainer
    trainer = pl.Trainer(accelerator="auto", devices="auto")

    # Run prediction
    # predictions = trainer.predict(model, dataloader, ckpt_path=CKPT_PATH)
    predictions = trainer.predict(model, data_module) # ckpt_path=CKPT_PATH

    print('trainer.predict done!!!')

    # Post-process predictions
    prediction_mask = torch.argmax(predictions[0], dim=0).cpu().numpy()

    # Load the original image for visualization
    with h5py.File(IMAGE_PATH, "r") as f:
        original_image = f["image"][:]

    # Visualize the results
    visualize_results(original_image, prediction_mask)

def run_prediction(model, test_loader, device):

    pred_probs = []
    ground_truths = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # inputs, labels, _ = data
            inputs = batch["image"].to(device) 
            labels = batch["label"].to(device)
            # if labels.ndim > 1:
            #     labels = labels.to(device).squeeze()
            # else:
            #     labels = labels.to(device)
            if labels.ndim == 0:
                basic.outputlogMessage(f"error: labels.ndim == 0, ndim: {labels.ndim}, size: {labels.size()}, value: {labels.tolist()}")
                continue

            outputs = model(inputs).output          
            pred_probs.append(outputs)    # .cpu().numpy()
            ground_truths.append(labels)             #.cpu().numpy()

    # pre_probs = np.concatenate(pre_probs, 0)     # for numpy array
    pred_probs = torch.cat(pred_probs, 0)               # for tensor
    ground_truths = torch.cat(ground_truths, 0)               # for tensor

    # pred_probs = []
    # ground_truths = []
    # with torch.no_grad():
    #     batch = next(iter(test_loader))
    #     images = batch["image"].to(device)
    #     outputs = model(images)
    #     # print(outputs)
    #     preds = outputs.output.argmax(dim=1)
    #     print(preds.shape, preds)
    # # pred_probs = np.concatenate(pred_probs, 0)     # for numpy array
    # pred_probs = torch.cat(pred_probs, 0)               # for tensor
    # ground_truths = torch.cat(ground_truths, 0)               # for tensor

    return pred_probs, ground_truths

def predict_remoteSensing_data_rsBigModel(para_file, area_idx, area_ini, area_save_dir,model_type, trained_model, batch_size=16):

    # run the prediction
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    data_transform = get_data_transforms()

    inf_extract_img_dir = parameters.get_directory_None_if_absence(para_file,'inf_extract_img_dir')
    if inf_extract_img_dir is not None:
        inf_extract_img_dir = os.path.join(inf_extract_img_dir, os.path.basename(area_save_dir))

    num_workers = parameters.get_digit_parameters(para_file, 'process_num', 'int')
    class_labels_txt = parameters.get_file_path_parameters(para_file,'class_labels')
    test_img_label_txt = prepare_dataset_for_classify(para_file, area_ini, area_save_dir, test=True, extract_img_dir=inf_extract_img_dir)
    
    data_module = RSPatchTxtModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train_txt=None,
        valid_txt=None,
        test_txt=test_img_label_txt, 
        label_txt=class_labels_txt,
        transforms=data_transform
    )
    data_module.setup("test")
    test_dataset = data_module.test_dataset
    num_classes = len(test_dataset.classes)
    print(f"Number of samples in the test dataset: {len(test_dataset)}")

    if len(test_dataset) < 1:
        print('No images for prediction')
        return None

    # load the trained model
    from terratorch.tasks import ClassificationTask
    # backbone is prithvi_eo_v2_300
    # model = ClassificationTask(
    #     model_factory="EncoderDecoderFactory",
    #     model_args={
    #         "backbone": model_type.lower(),  # use the prithvi_eo_v2_300 backbone
    #         "backbone_pretrained": False,    # load pretrained weights
    #         "backbone_bands": ["BLUE", "GREEN", "RED"],
    #         "decoder": "IdentityDecoder",   # no decoder is used
    #         "head_dropout": 0.1,            # dropout in the classification head
    #         "head_dim_list": [384, 128],    # hidden dimension of the head
    #         "num_classes": num_classes,              #
    #     }
    # )
    model = ClassificationTask.load_from_checkpoint(trained_model)
    model.to(device)

    test_loader = data_module.test_dataloader()
    pre_probs, ground_truths = run_prediction(model, test_loader, device)


    save_path = os.path.join(area_save_dir, os.path.basename(area_save_dir)+'-classify_results.json' )
    save_k = min(5, len(test_dataset.classes))
    res_dict, _ = save_prediction_results(test_dataset,pre_probs, save_path, k=save_k)

    top_probs_1, top_labels_1 = pre_probs.cpu().topk(1, dim=-1)

    # output accuracy
    # top1 accuracy
    top1_acc_save_path = os.path.join(area_save_dir, 'top1_accuracy.txt' )
    calculate_top_k_accuracy(top_labels_1, ground_truths, save_path=top1_acc_save_path, k=1)

    

    # move selection of random samples into prediction step (because after prediciton, these images will be removed)
    class_ids_for_manu_check = parameters.get_string_list_parameters_None_if_absence(para_file,'class_ids_for_manu_check')
    if class_ids_for_manu_check is not None:
        class_ids_for_manu_check = [ int(item) for item in class_ids_for_manu_check]
        sel_count = parameters.get_digit_parameters(para_file,'sample_num_per_class','int')
        class_labels_txt = parameters.get_file_path_parameters(para_file,'class_labels')
        class_id_dict = read_label_ids_local(class_labels_txt)
        image_path_list = test_dataset.img_list
        for c_id in class_ids_for_manu_check:
            select_sample_for_manu_check(c_id, area_save_dir, sel_count, class_id_dict, image_path_list, res_dict)

    # remove extracted images after prediction, to release disk space
    b_rm_extracted_subImage = parameters.get_bool_parameters_None_if_absence(para_file,'b_rm_extracted_subImage')
    if b_rm_extracted_subImage is True:
        for img_p in test_dataset.img_list:
            io_function.delete_file_or_dir(img_p)


def classify_one_region_rsBigModel(area_idx, area_ini, para_file, area_save_dir, gpuid, inf_list_file, model_type, trained_model):
    
    inf_batch_size = parameters.get_digit_parameters(para_file,'inf_batch_size','int')

    done_indicator = '%s_done'%inf_list_file
    if os.path.isfile(done_indicator):
        basic.outputlogMessage('warning, %s exist, skip prediction'%done_indicator)
        return
    # use a specific GPU for prediction, only inference one image
    time0 = time.time()
    if gpuid is not None:
        #TODO: this doesn't work after torch already be imported,
        # can work if set CUDA_VISIBLE_DEVICES in the shell
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    predict_remoteSensing_data_rsBigModel(para_file, area_idx, area_ini, area_save_dir,model_type, trained_model, batch_size=inf_batch_size)

    duration = time.time() - time0
    # os.system('echo "$(date): time cost of inference for image in %s: %.2f seconds">>"time_cost.txt"' % (inf_list_file, duration))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{now_str}: time cost of inference for image in {inf_list_file}: {duration:.2f} seconds"
    io_function.append_text_to_file("time_cost.txt",line)

    # write a file to indicate that the prediction has done.
    io_function.append_text_to_file(done_indicator,f"{inf_list_file}")
    # os.system('echo %s > %s_done'%(inf_list_file,inf_list_file))


def parallel_prediction_main(para_file, trained_model, task_type):

    print(f"run prediction {task_type} using the trained model: {trained_model}") # (run parallel if using multiple GPUs)
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
    used_gpu_ids = []
    for area_idx, area_ini in enumerate(multi_inf_regions):

        inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')
        inf_image_patch_labels_txt = parameters.get_file_path_parameters_None_if_absence(area_ini,
                                                'inf_image_patch_labels')

        if inf_image_patch_labels_txt is not None:
            print('Parameter: inf_image_patch_labels_txt  is set, will read data from it for prediction')
            # img_label_list = io_function.read_list_from_txt(inf_image_patch_labels_txt)
        else:
            # it is ok consider a file name as pattern and pass it the following functions to get file list
            inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')
            inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir, inf_image_or_pattern)
            img_count = len(inf_img_list)
            if img_count < 1:
                raise ValueError(
                    'No image for inference, please check inf_image_dir (%s) and inf_image_or_pattern (%s) in %s'
                    % (inf_image_dir, inf_image_or_pattern, area_ini))

        area_name_remark_time = parameters.get_area_name_remark_time(area_ini)
        area_save_dir = os.path.join(outdir, area_name_remark_time)
        inf_list_file = os.path.join(area_save_dir, '%d.txt' % area_idx)

        done_indicator = '%s_done' % inf_list_file
        if os.path.isfile(done_indicator):
            basic.outputlogMessage('warning, %s exist, skip prediction' % done_indicator)
            continue

        # parallel inference images for this area

        while basic.alive_process_count(sub_tasks) >= maximum_prediction_jobs:
            print(datetime.now(),
                  '%d jobs are running simultaneously, wait 30 seconds' % basic.alive_process_count(sub_tasks))
            time.sleep(30)  # wait 30 seconds, then check the count of running jobs again

        if b_use_multiGPUs:
            deviceIDs = bim_utils.get_wait_available_GPU(machine_name,check_every_sec=30)

            # in the case it takes long time to prepare data, tasks need to make sure task equally distributed to different gpus
            gpu_task_count = [used_gpu_ids.count(id) for id in deviceIDs]
            min_loc = gpu_task_count.index(min(gpu_task_count))
            gpuid = deviceIDs[min_loc]

            basic.outputlogMessage('%d: predict region: %s on GPU %d of %s' % (area_idx, area_name_remark_time, gpuid, machine_name))
            used_gpu_ids.append(gpuid)
        else:
            gpuid = None
            basic.outputlogMessage('%d: predict region: %s on %s' % (area_idx, area_name_remark_time, machine_name))

        # if it already exists, then skip
        if os.path.isdir(area_save_dir) and bim_utils.is_file_exist_in_folder(area_save_dir):
            basic.outputlogMessage('folder of %dth region (%s) already exist, '
                                   'it has been predicted or is being predicted' % (area_idx, area_name_remark_time))
            continue

        # run inference
        io_function.mkdir(area_save_dir)

        with open(inf_list_file, 'w') as inf_obj:
            inf_obj.writelines(area_name_remark_time + '\n')

        if task_type == 'classification':
            sub_process = Process(target=classify_one_region_rsBigModel,
                              args=(area_idx, area_ini, para_file, area_save_dir, gpuid, inf_list_file, model_type, trained_model))
        elif task_type == 'segmentation':
            print('Segmentation is not implemented yet.')
        else:
            print('Error: unsupported task type: {}'.format(task_type))
            return False

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
            file_exist = os.path.isdir(area_save_dir) and bim_utils.is_file_exist_in_folder(area_save_dir)
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
        time.sleep(10)

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
    out_str = "%s: time cost of total parallel inference on %s: %d seconds (%.f hours)" % (
        str(end_time), machine_name, diff_time.total_seconds(), diff_time.total_seconds()/3600.0 )
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

    pass

def test_parallel_prediction_main():
    # change dir
    WORK_DIR= "/home/hlc/Data/slump_demdiff_classify/cnn_rsModel_classify"
    os.chdir(WORK_DIR)
    para_file = 'main_para_exp15.ini'
    trained_model = 'exp15_bak/best-val-Prithvi_EO_V2_300.ckpt'
    task_type = 'classification'
    parallel_prediction_main(para_file,trained_model,task_type)


def main(options, args):

    para_file = args[0]
    trained_model = options.trained_model

    if len(trained_model) > 1 and os.path.isfile(trained_model) is False:
        raise IOError(f'trained_model: {trained_model} is set but do not exist, please check. Note ~, *, ? are not supported')

    task_type = options.task_type

    parallel_prediction_main(para_file,trained_model,task_type)

if __name__ == "__main__":

    # test_parallel_prediction_main()
    # sys.exit(0)

    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2026-03-26")
    parser.description = 'Introduction: run the prediction using the RS big models with custom data'

    parser.add_option("-m", "--trained_model",
                      action="store", dest="trained_model",default='',
                      help="the trained model")
    
    parser.add_option("", "--task_type",
                      action="store", dest="task_type",default='',
                      help="the task type, should be one of 'classification', 'segmentation', etc ")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
