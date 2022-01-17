import os.path as osp

# from .builder import DATASETS
# from .custom import CustomDataset

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import os,sys

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import basic_src.io_function as io_function
import split_image
import raster_io

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
            # super(RSImagePatches, self).__init__(
            #     img_suffix='.png', seg_map_suffix='.png', split=split, test_mode=test_mode, **kwargs)

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


if __name__ == '__main__':
    pass