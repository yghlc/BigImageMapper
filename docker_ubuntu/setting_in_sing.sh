#!/bin/bash

# show environment
echo "HOME: " $HOME
echo "PATH: " $PATH
echo "LD_LIBRARY_PATH: " $LD_LIBRARY_PATH

# however, it turns out it not necessary install  miniconda and the python packages inside container,
# tye can also be install outside the container, which is more convenient

# install miniconda (different machine, the path can change, so need to install it again)
#wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
#sh Miniconda2-latest-Linux-x86_64.sh -p $HOME/programs/miniconda2 -b

# install OTB-6.6.1-Linux64.run (different machine, the path can change, so need to install it again)
#sh OTB-6.6.1-Linux64.run --accept --target ${HOME}/programs/OTB-6.6.1-Linux64
#ln -s ${HOME}/programs/miniconda2/lib/libpython2.7.so.1.0 ${HOME}/programs/OTB-6.6.1-Linux64/lib/libpython2.7.so.1.0

which python
#ls /home/hlc/
which cp_shapefile
echo $TZ
cat /etc/timezone
date

#cat /etc/environment
#cat /etc/locale.gen
#cat /etc/locale.conf


## install tensorflow, the same as cryo03 
pip install tensorflow-gpu==1.6

## install gdal
conda install gdal=2.3

## install  rasterio
pip install rasterio

## for model shapefile
pip install pyshp==1.2.12

pip install rasterstats

# for PIL
pip install pillow

# for image augmentation
pip install imgaug
