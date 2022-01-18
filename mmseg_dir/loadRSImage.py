#!/usr/bin/env python
# Filename: loadRSImage.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 January, 2022
"""

import os.path as osp

import os,sys
deeplabforRS = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import raster_io

from mmseg.datasets.builder import PIPELINES
import numpy as np

@PIPELINES.register_module()
class LoadRSImagePatch(object):
    """Load an sub-image from file of a remote sensing image.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        # self.color_type = color_type
        # self.file_client_args = file_client_args.copy()
        # self.file_client = None
        # self.imdecode_backend = imdecode_backend
        print('__init__ of LoadRSImagePatch')

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)
        print('__call__ of LoadRSImagePatch','try to read a image patch',results)
        img_id, org_img,boundary, patch_idx = results['img_id'], results['org_img'],results['boundary'], results['patch_idx']
        img, nodata = raster_io.read_raster_all_bands_np(org_img, boundary=boundary)
        if nodata is not None:
            img[np.where(img == nodata)] = 0  # replace nodata as 0

        if self.to_float32:
            img = img.astype(np.float32)

        results['out_file'] = 'I%d_%d.tif' % (img_id, patch_idx) # # save file name?
        results['boundary'] = boundary
        results['filename'] = osp.join('I%d'%img_id,'I%d_%d.tif'%(img_id,patch_idx))
        results['ori_filename'] = org_img # results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        print('__repr__ of LoadRSImagePatch')
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

if __name__ == '__main__':
    pass