import os.path as osp

# from .builder import DATASETS
# from .custom import CustomDataset

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from mmseg.datasets.builder import PIPELINES

import os,sys

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import basic_src.io_function as io_function
import split_image
import raster_io
import numpy as np

# copy from build_RS_data.py
class patchclass(object):
    """
    store the information of each patch (a small subset of the remote sensing images)
    """
    def __init__(self,org_img,boundary):
        self.org_img = org_img  # the original remote sensing images of this patch
        self.boundary=boundary      # the boundary of patch (xoff,yoff ,xsize, ysize) in pixel coordinate
    def boundary(self):
        return self.boundary

@DATASETS.register_module()
class RSImagePatches(CustomDataset):
    """Remote sensing dataset for thaw slump.

    Args:
        split (str): Split txt file for Remote sensing patches.
    """

    CLASSES = ('background', 'thawslump')

    PALETTE = [[0, 0, 0], [0, 0, 128]]

    def __init__(self, split,test_mode=False,rsimage='',rsImg_id=0,tile_width=480,tile_height=480,
                 overlay_x=160,overlay_y=160, **kwargs):
        """
        get patches of a remote sensing images for predictions, in the training part, still using the MMSeg default loader
        Only handle one remote sensing images

        Args:
            split: split
            test_mode: test or not
            rsimage: a list containing the path of a remote sensing image
            rsImg_id: the id (int) of the input rsimage
            tile_width: width of the patch
            tile_height: height of the patch
            overlay_x: adjacent overlap in X directory
            overlay_y: adjacent overlap in Y directory
            **kwargs:
        """
        if test_mode:
            super(RSImagePatches, self).__init__(
                img_suffix='.png', seg_map_suffix='.png', split=None, test_mode=test_mode, **kwargs)

            # initialize the images
            assert len(rsimage) > 1 and osp.exists(rsimage)
            patches_of_a_image = self.get_an_image_patches(rsimage, tile_width, tile_height, overlay_x, overlay_y)
            self.img_infos = [ {'img_id':rsImg_id, 'patch_idx':idx, 'patch_obj':patch }
                               for idx, patch in enumerate(patches_of_a_image)]

        else:
            super(RSImagePatches, self).__init__(
                img_suffix='.png', seg_map_suffix='.png', split=split, test_mode=test_mode, **kwargs)
            # check image dir's existence
            assert osp.exists(self.img_dir) and self.split is not None

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            # need something else
            return self.img_infos[idx]
        else:
            return self.prepare_train_img(idx)

    def get_an_image_patches(self,image_path,tile_width,tile_height,overlay_x,overlay_y):

        height, width, b_count, dtypes = raster_io.get_height_width_bandnum_dtype(image_path)
        # split the image
        patch_boundary = split_image.sliding_window(width, height, tile_width, tile_height,
                                                    overlay_x, overlay_y)
        patches_of_a_image = []
        for t_idx, patch in enumerate(patch_boundary):
            img_patch = patchclass(image_path, patch)
            patches_of_a_image.append(img_patch)
        return patches_of_a_image


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

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)
        img_id, patch_obj, patch_idx = results['img_id'], results['patch_obj'], results['patch_idx']
        img, nodata = raster_io.read_raster_all_bands_np(patch_obj.org_img, boundary=patch_obj.boundary)
        if nodata is not None:
            img[np.where(img == nodata)] = 0  # replace nodata as 0

        if self.to_float32:
            img = img.astype(np.float32)

        results['out_file'] = 'I%d_%d.tif' % (img_id, patch_idx) # # save file name?
        results['boundary'] = patch_obj.boundary
        results['filename'] = osp.join('I%d'%img_id,'I%d_%d.tif'%(img_id,patch_idx))
        results['ori_filename'] = patch_obj.org_img # results['img_info']['filename']
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
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



if __name__ == '__main__':
    pass