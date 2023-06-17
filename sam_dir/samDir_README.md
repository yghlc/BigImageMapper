## SAM: segment Anything model
Using [SAM](https://github.com/facebookresearch/segment-anything) for segmenting landforms 
from remote sensing imagery.

## Install
Please following the instruction at [SAM](https://github.com/facebookresearch/segment-anything) 
to install Python and create a conda environment (name: pytorch). 

```
# In the conda environment, install the forked version
git clone https://github.com/yghlc/segment-anything-largeImage
git clone git@github.com:yghlc/segment-anything-largeImage.git
cd segment-anything;
pip install -e .

# other packages
conda install -c conda-forge rasterio
pip install scikit-image
pip install GPUtil
pip install opencv-python pycocotools matplotlib onnxruntime onnx

```

Other dependencies please follow the main readme file to install.

## How to use
