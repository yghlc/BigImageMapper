# Setting and running this repo

This file introduces how to setting the environment for this reso on the ITSC GPU clusters. 
It also provides a reference for setting and running on your own workstations.

## Setting on the ITSC GPU cluster
Before the setting, it is good to know the [policies](https://www.cuhk.edu.hk/itsc/hpc/policies.html) and 
[storage](https://www.cuhk.edu.hk/itsc/hpc/storage.html). It also requires users to 
submit a job for running computing script using [Slurm](https://www.cuhk.edu.hk/itsc/hpc/slurm.html). 
[Singularity](https://www.cuhk.edu.hk/itsc/hpc/singularity.html) can create a isolated software environment for running scripts. 

Due to the small size of home folder (20 GB), data should be stored in other places. 

#### Step 1: copy a singularity image
For easy use and convenience, I have built a singularity image and uploaded it to Dropbox. 
After logining to the ITSC server, to download the image, run:

    wget https://www.dropbox.com/s/pu95hkn93tmhx05/ubuntu16.04_itsc_tf.simg?dl=0 --output-document=ubuntu16.04_itsc_tf.simg

#### Step 2: install dependencies
We install all the dependencies under the home folder. Because ITSC uses the group
information to manage the storage, to reduce the storage of home folder, we change 
the group information to *LinLiu*. As the following commands:

    mkdir packages
    chgrp -R LinLiu packages

Install CUDA 9.0 and cuDNN 7.0, these are required by tensorflow 1.6. 
These should be downloaded via NVIDIA website, but for this tutorial, you can download them
from my Dropbox. 
    
    wget https://www.dropbox.com/s/1bi3udi48dsw2c1/cuda-9.0.tar.gz?dl=0 --output-document=cuda-9.0.tar.gz 
    wget https://www.dropbox.com/s/2v4sfdjbsgwzi1t/cuDNN_7.0.tar.gz?dl=0 --output-document=cuDNN_7.0.tar.gz

Then we unpackage these to the folder "programs" 
 
    mkdir -p programs
    tar xvf cuDNN_7.0.tar.gz -C programs
    tar xvf cuda-9.0.tar.gz -C programs

DeepLabv3+ provides some [pre-trained model](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md), 
in this tutorial, we use one of them and download it to *Data*

    mkdir -p ./Data/deeplab/v3+/pre-trained_model
    wget https://www.dropbox.com/s/0h7g5cjyvxywkt1/deeplabv3_xception_2018_01_04.tar.gz?dl=0 --output-document=./Data/deeplab/v3+/pre-trained_model/deeplabv3_xception_2018_01_04.tar.gz

Download a script which may be used in some of the codes, and make it executable.

    mkdir -p ./bin
    wget https://www.dropbox.com/s/6sdwu3tx9jwzfsm/cp_shapefile?dl=0 --output-document=./bin/cp_shapefile
    chmod a+x ./bin/cp_shapefile

Clone codes from GitHub:

    git clone https://github.com/yghlc/DeeplabforRS.git ./codes/PycharmProjects/DeeplabforRS
    git clone https://github.com/yghlc/Landuse_DL ./codes/PycharmProjects/Landuse_DL
    git clone https://github.com/yghlc/models.git ./codes/PycharmProjects/tensorflow/yghlc_tf_model


Install python using miniconda 

    wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
    sh Miniconda2-latest-Linux-x86_64.sh -p $HOME/programs/miniconda2 -b

    
Install tensorflow 1.6 (a relative old version) and other python packages. <!-- The installation will run inside 
the container, so we need to submit a job for running singularity. -->
    
    ${HOME}/programs/miniconda2/bin/pip install tensorflow-gpu==1.6
    ${HOME}/programs/miniconda2/bin/conda install gdal=2.3
    ${HOME}/programs/miniconda2/bin/pip install rasterio
    ${HOME}/programs/miniconda2/bin/pip install pyshp==1.2.12
    ${HOME}/programs/miniconda2/bin/pip install rasterstats
    ${HOME}/programs/miniconda2/bin/pip install pillow
    ${HOME}/programs/miniconda2/bin/pip install imgaug

Because some of the sub-folders don't change the group info to *LinLiu* or *LinLiuScratch*, we modify them again.
    
    chgrp -R LinLiu programs
    chgrp -R LinLiuScratch Data
    chgrp -R LinLiu codes

To inquire the storage quota of you home folder, please run:
    
    lfs quota -gh username /lustre    #  replace username as your user name



 <!--We need to run our scripts inside a singularity container by submitting jobs. Copy a slurm example to current folder, 

    cp ~/codes/PycharmProjects/Landuse_DL/docker_ubuntu1604/singularity.sh .

Copy an example for running scripts inside the singularity container to current folder:
    
    cp ~/codes/PycharmProjects/Landuse_DL/docker_ubuntu1604/run_INsingularity_miniconda.sh .
 
 -->
  
 <!-- on ITSC server, I failed to set "HOME" inside the singularity, 
 maybe we remove "packages" and use the HOME of the host machine.
   -->

## Run training, prediction, and post-processing
After the environment for running Landuse_DL is ready, you can start training your model as well as prediction. 
Suppose your working folder is *test_deeplabV3+_1*, in this folder, a list of files should be presented:
    
    exe.sh
    para.ini
    inf_image_list.txt
    run_INsingularity_miniconda.sh 
    singularity.sh

*exe.sh* is a script for running the whole process, including preparing training images, 
training, prediction, and post-processing. You may want to comment out some of the lines in this file 
if you just want to run a few steps. This file can be copied from *Landuse_DL/thawslumpScripts/exe_qtp.sh*.
In this file, you may also need to some input file, such as change *para_qtp.ini* to *para.ini*, 
comment out lines related to *PATH* and *CUDA_VISIBLE_DEVICES*, and the value of *gpu_num*.


*para.ini* is the file where you define input files and parameters. Please edit it accordingly. 
An example of *para.ini* is available at *Landuse_DL/thawslumpScripts/para_qtp.ini*.

*inf_image_list.txt* stores the image file names for prediction as follows,

    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000000000_8bit_Albers.tif
    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000026880_8bit_Albers.tif
    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000053760_8bit_Albers.tif
    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000080640_8bit_Albers.tif
    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000107520_8bit_Albers.tif

*run_INsingularity_miniconda.sh* is need to run the script inside the [Singularity](https://www.cuhk.edu.hk/itsc/hpc/singularity.html) container. 
Modify it according if you want to run other script. *Landuse_DL/docker_ubuntu1604/run_INsingularity_miniconda.sh*
 is an example. 

    exe_script=./exe.sh

*singularity.sh* is for submitting a job. Please also modify it accordingly. Please refer to ITSC
website ([Slurm](https://www.cuhk.edu.hk/itsc/hpc/slurm.html)) for details. 
*Landuse_DL/docker_ubuntu1604/singularity.sh* is an example. 
Running the following script for submiting a job. 
    
    sbatch singularity.sh






