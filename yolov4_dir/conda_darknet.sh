#!/bin/bash

# install conda environment for running training and prediction of darknet (YOLO)

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 10 April, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


conda create -n darknet python=3.7

conda activate darknet

pip install numpy psutil GPUtil openpyxl xlsxwriter \
&& pip install opencv-python rasterio geopandas  \
&& pip install pyshp==1.2.12 rasterstats \
&& pip install scikit-image scikit-learn \
&& pip install pytest

conda install -y gdal



