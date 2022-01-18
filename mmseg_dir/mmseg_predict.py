#!/usr/bin/env python
# Filename: mmseg_predict.py 
"""
introduction: run prediction of a deep learning model for semantic segmentation using MMSegmentation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 January, 2022
"""

import os,sys
import os.path as osp
from optparse import OptionParser
from datetime import datetime
import time
import GPUtil
# from multiprocessing import Process

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.raster_io as raster_io

import mmcv
import torch
from torch.multiprocessing import Process,set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

code_dir2 = os.path.expanduser('~/codes/PycharmProjects/yghlc_mmsegmentation')
sys.path.insert(0, code_dir2)

import mmseg
print('mmseg_path:',mmseg.__path__)
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.image import tensor2imgs

import numpy as np

# open-mmlab models
# from mmcv.utils import Config

# if we use mmseg in yghlc_mmsegmentation, then don't need to import these two
# from RSImageMMSeg import RSImagePatches    # register the dataset, also added to mmseg/datasets/__init__.py
# from loadRSImage import LoadRSImagePatch   # register the dataset, also added to mmseg/datasets/pipelines/__init__.py


def is_file_exist_in_folder(folder):
    # just check if the folder is empty
    if len(os.listdir(folder)) == 0:
        return False
    else:
        return True


def single_gpu_prediction_rsImage(model,data_loader,out_dir=None):
    model.eval()
    # results = []
    dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    patch_count = len(dataset)

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for img, img_meta, res in zip(imgs, img_metas,result):
            # h, w, _ = img_meta['img_shape']
            # img_show = img[:h, :w, :]
            #
            # ori_h, ori_w = img_meta['ori_shape'][:-1]
            # img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            save_path = osp.join(out_dir,img_meta['filename'])
            org_img = img_meta['ori_filename']
            boundary = img_meta['boundary']
            # save the patch
            print('saving patch:%s results of %s, total: %d'%(img_meta['filename'],osp.basename(org_img),patch_count))
            res_8bit = res.astype(dtype=np.uint8)
            raster_io.save_numpy_array_to_rasterfile(res_8bit,save_path,org_img,boundary=boundary,verbose=False)

        # results.extend(result)

        # batch_size = len(result)
        # for _ in range(batch_size):
        #     prog_bar.update()

    # return results
    return True

def predict_rsImage_mmseg(config_file,trained_model,image_path, img_save_dir,batch_size=1,
                          tile_width=480, tile_height=480, overlay_x=160, overlay_y=160):
    cfg = mmcv.Config.fromfile(config_file)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    distributed = False

    # test_mode=False,rsimage='',rsImg_id=0,tile_width=480,tile_height=480,
    #                  overlay_x=160,overlay_y=160

    data_args ={'rsImg_predict':True,'rsimage':image_path,'tile_width':tile_width,'tile_height':tile_height,
                'overlay_x':overlay_x,'overlay_y':overlay_y}
    dataset = build_dataset(cfg.data.test,default_args=data_args)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, trained_model, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    # no distributed
    model = MMDataParallel(model, device_ids=[0])
    single_gpu_prediction_rsImage(model,data_loader, img_save_dir)




def predict_one_image_mmseg(para_file, image_path, img_save_dir, inf_list_file, gpuid,trained_model):
    """ run prediction of one image
    """
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    base_config_file = parameters.get_string_parameters(network_ini, 'base_config')
    config_file = osp.basename(io_function.get_name_by_adding_tail(base_config_file, expr_name))

    inf_batch_size = parameters.get_digit_parameters(network_ini,'inf_batch_size','int')

    patch_width = parameters.get_digit_parameters(para_file,'inf_patch_width','int')
    patch_height = parameters.get_digit_parameters(para_file,'inf_patch_height','int')
    adj_overlay_x = parameters.get_digit_parameters(para_file,'inf_pixel_overlay_x','int')
    adj_overlay_y = parameters.get_digit_parameters(para_file,'inf_pixel_overlay_y','int')

    done_indicator = '%s_done'%inf_list_file
    if os.path.isfile(done_indicator):
        basic.outputlogMessage('warning, %s exist, skip prediction'%done_indicator)
        return
    if os.path.isdir(img_save_dir) is False:
        io_function.mkdir(img_save_dir)
    # use a specific GPU for prediction, only inference one image
    time0 = time.time()
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    predict_rsImage_mmseg(config_file, trained_model, image_path, img_save_dir, batch_size=inf_batch_size,
                          tile_width=patch_width, tile_height=patch_height, overlay_x=adj_overlay_x, overlay_y=adj_overlay_y)

    duration = time.time() - time0
    os.system('echo "$(date): time cost of inference for image in %s: %.2f seconds">>"time_cost.txt"' % (inf_list_file, duration))
    # write a file to indicate that the prediction has done.
    os.system('echo %s > %s_done'%(inf_list_file,inf_list_file))

    return


def mmseg_parallel_predict_main(para_file,trained_model):

    print("MMSegmetation prediction using the trained model (run parallel if use multiple GPUs)")
    machine_name = os.uname()[1]
    start_time = datetime.now()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    # network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    # mmseg_repo_dir = parameters.get_directory(network_ini, 'mmseg_repo_dir')
    # mmseg_code_dir = osp.join(mmseg_repo_dir,'mmseg')

    # if os.path.isdir(mmseg_code_dir) is False:
    #     raise ValueError('%s does not exist' % mmseg_code_dir)

    # # set PYTHONPATH to use my modified version of mmseg
    # if os.getenv('PYTHONPATH'):
    #     os.environ['PYTHONPATH'] = os.getenv('PYTHONPATH') + ':' + mmseg_code_dir
    # else:
    #     os.environ['PYTHONPATH'] = mmseg_code_dir
    # print('\nPYTHONPATH is: ',os.getenv('PYTHONPATH'))


    if trained_model is None:
        trained_model = os.path.join(expr_name, 'latest.pth')

    outdir = parameters.get_directory(para_file, 'inf_output_dir')
    # remove previous results (let user remove this folder manually or in exe.sh folder)
    io_function.mkdir(outdir)

    # get name of inference areas
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')
    b_use_multiGPUs = parameters.get_bool_parameters(para_file, 'b_use_multiGPUs')

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
            raise ValueError('No image for inference, please check inf_image_dir and inf_image_or_pattern in %s' % area_ini)

        area_save_dir = os.path.join(outdir, area_name + '_' + area_remark + '_' + area_time)
        io_function.mkdir(area_save_dir)

        # parallel inference images for this area
        CUDA_VISIBLE_DEVICES = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            CUDA_VISIBLE_DEVICES = [int(item.strip()) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        idx = 0
        while idx < img_count:

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
                basic.outputlogMessage('%d: predict image %s on GPU %d of %s' % (idx, inf_img_list[idx], gpuid, machine_name))
            else:
                gpuid = None
                basic.outputlogMessage('%d: predict image %s on %s' % (idx, inf_img_list[idx], machine_name))

            # run inference
            img_save_dir = os.path.join(area_save_dir, 'I%d' % idx)
            inf_list_file = os.path.join(area_save_dir, '%d.txt' % idx)

            done_indicator = '%s_done' % inf_list_file
            if os.path.isfile(done_indicator):
                basic.outputlogMessage('warning, %s exist, skip prediction' % done_indicator)
                idx += 1
                continue

            # if it already exist, then skip
            if os.path.isdir(img_save_dir) and is_file_exist_in_folder(img_save_dir):
                basic.outputlogMessage('folder of %dth image (%s) already exist, '
                                       'it has been predicted or is being predicted' % (idx, inf_img_list[idx]))
                idx += 1
                continue

            with open(inf_list_file, 'w') as inf_obj:
                inf_obj.writelines(inf_img_list[idx] + '\n')

            sub_process = Process(target=predict_one_image_mmseg,
                                  args=(para_file,inf_img_list[idx], img_save_dir, inf_list_file,
                                        gpuid, trained_model))
            sub_process.start()
            sub_tasks.append(sub_process)

            if b_use_multiGPUs is False:
                # wait until previous one finished
                while sub_process.is_alive():
                    time.sleep(1)

            idx += 1

            # wait until predicted image patches exist or exceed 20 minutes
            time0 = time.time()
            elapsed_time = time.time() - time0
            while elapsed_time < 20 * 60:
                elapsed_time = time.time() - time0
                file_exist = os.path.isdir(img_save_dir) and is_file_exist_in_folder(img_save_dir)
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
    str(end_time), machine_name, diff_time.seconds)
    basic.outputlogMessage(out_str)
    with open("time_cost.txt", 'a') as t_obj:
        t_obj.writelines(out_str + '\n')



def main(options, args):

    para_file = args[0]
    trained_model = options.trained_model

    mmseg_parallel_predict_main(para_file,trained_model)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2022-01-14")
    parser.description = 'Introduction: run prediction in parallel on MMSegmentation '

    parser.add_option("-m", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model for prediction")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)







