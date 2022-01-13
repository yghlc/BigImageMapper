import os.path as osp

# from .builder import DATASETS
# from .custom import CustomDataset

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class RSImagePatches(CustomDataset):
    """Remote sensing dataset for thaw slump.

    Args:
        split (str): Split txt file for Remote sensing patches.
    """

    CLASSES = ('background', 'thawslump')

    PALETTE = [[0, 0, 0], [0, 0, 128]]

    def __init__(self, split, **kwargs):
        super(RSImagePatches, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


if __name__ == '__main__':
    pass