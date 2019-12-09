# Setting and running this repo

This file introduce how to setting this environment on the ITSC GPU clusters. 
It also provides a reference for setting and running this repo in your own workstation.

## Setting on the ITSC GPU cluster
Before the setting, it is good to know the [policies](https://www.cuhk.edu.hk/itsc/hpc/policies.html) and 
[storage](https://www.cuhk.edu.hk/itsc/hpc/storage.html). It also requires users to 
submit a job for running computing script using [Slurm](https://www.cuhk.edu.hk/itsc/hpc/slurm.html). 
[Singularity](https://www.cuhk.edu.hk/itsc/hpc/singularity.html) can create a isolated software environment for running scripts. 

Due to small size of home folder (20 GB), data should be stored in other places. 

#### Step 1: copy a singularity image
For easy use and convenience, I have built a singularity image and uploaded it to Dropbox. 
After logining to the ITSC server, to download the image, run:

    wget https://www.dropbox.com/s/pu95hkn93tmhx05/ubuntu16.04_itsc_tf.simg?dl=0 --output-document=ubuntu16.04_itsc_tf.simg

#### Step 2: install dependencies
We create a folder named "packages" under the home folder. Because ITSC uses the group
information to manage the storage, to reduce the storage of home folder, we change 
the group infomation to "LinLiu". As the following commands:

    mkdir packages
    chgrp -R LinLiu packages

Install CUDA 9.0 and cuDNN 7.0, these are required by tensorflow 1.6. 
These should be downloaded via NVIDIA website, but for this tutorial, you can download them
from my Dropbox. 
    
    wget https://www.dropbox.com/s/1bi3udi48dsw2c1/cuda-9.0.tar.gz?dl=0 --output-document=packages/cuda-9.0.tar.gz 
    wget https://www.dropbox.com/s/2v4sfdjbsgwzi1t/cuDNN_7.0.tar.gz?dl=0 --output-document=packages/cuDNN_7.0.tar.gz

Then we unpackage these to the folder "packages/programs" 
 
    mkdir -p packages/programs
    tar xvf packages/cuDNN_7.0.tar.gz -C packages/programs
    tar xvf packages/cuda-9.0.tar.gz -C packages/programs

DeepLabv3+ provides some [pre-trained model](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md), 
in this tutorial, we use one of them and download it to "packages"

    mkdir -p ./packages/Data/deeplab/v3+/pre-trained_model
    wget https://www.dropbox.com/s/0h7g5cjyvxywkt1/deeplabv3_xception_2018_01_04.tar.gz?dl=0 --output-document=./packages/Data/deeplab/v3+/pre-trained_model/deeplabv3_xception_2018_01_04.tar.gz

Download a script which may be used in some of the codes, and make it executable.

    mkdir -p ./packages/bin
    wget https://www.dropbox.com/s/6sdwu3tx9jwzfsm/cp_shapefile?dl=0 --output-document=./packages/bin/cp_shapefile
    chmod a+x ./packages/bin/cp_shapefile

Clone codes from GitHub:

    git clone https://github.com/yghlc/DeeplabforRS.git ./packages/codes/PycharmProjects/DeeplabforRS
    git clone https://github.com/yghlc/Landuse_DL ./packages/codes/PycharmProjects/Landuse_DL
    git clone https://github.com/yghlc/models.git ./packages/codes/PycharmProjects/tensorflow/yghlc_tf_model

Because some of the sub-folders don't change the group info to "LinLiu", we modify them again.
    
    chgrp -R LinLiu packages
    



## How to use
See the script: thawslumpScripts/exe.sh

## Contributions
Please let me know or pull a request if you spot any bug or typo. Thanks!
Any enhancement or new functions are also welcome!

## updates

## Dependencies and setting:
    Python package: Numpy, rasterio, GDAL 2.3, tensorflow-gpu 1.6, pyshp 1.2.12, pillow, imgaug
    Other: GDAL, OTB, ASP, CUDA 9.0, cudnn 7.0.
    More information on the setting can be found in 'docker_ubuntu1604/run_INsingularity_hlctest.sh'
    

## Disclaimer


## TODO




