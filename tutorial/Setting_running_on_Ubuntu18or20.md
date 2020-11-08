# Setting and running this repo


This file introduces how to set the environment for this repo on the Ubuntu 18 or 20 GPU machine.
It also provides a reference for setting and running on your own workstations.


#### Step 1: install dependencies
We install all the dependencies under the home folder. 

Install CUDA 10.0 and cuDNN 7.4, these are required by tensorflow 1.14. \
Ubuntu 18 or 20 already has CUDA 10.0 or 10.2 installed, so we cannot tensorflow 1.6 which requires CUDA 9.0.
They should be downloaded via NVIDIA website, but for this tutorial, you can download them
from my Dropbox. 
    
    wget https://www.dropbox.com/s/c1zzpriid1zng2d/cuda-10.0.tar.gz?dl=0 --output-document=cuda-10.0.tar.gz
    wget https://www.dropbox.com/s/8yj8fu7a08f7dx8/cuDNN_7.4_cuda10.tar.gz?dl=0 --output-document=cuDNN_7.4_cuda10.tar.gz

Then we unpack them to the folder "programs"
 
    mkdir -p programs
    tar xvf cuDNN_7.4_cuda10.tar.gz -C programs
    tar xvf cuda-10.0.tar.gz -C programs

Also, set the PATH and LD_LIBRARY_PATH in .bashrc, for example:
    
    # CUDA 10.0
    export PATH="HOME/programs/cuda-10.0/bin:$PATH"
    export LD_LIBRARY_PATH="HOME/programs/cuda-10.0/lib64:$LD_LIBRARY_PATH"
    #lib path of cuDNN v.7.4 for CUDA 10 (tf v1.14 need this one)
    export LD_LIBRARY_PATH="HOME/programs/cuDNN_7.4_cuda10/cuda/lib64:$LD_LIBRARY_PATH"
    
    noted: replace the "HOME" as your home folder, e.g., /home/hlc


DeepLabv3+ provides some [pre-trained model](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md), 
in this tutorial, we use one of them and download it to *Data*

    mkdir -p ./Data/deeplab/v3+/pre-trained_model
    wget https://www.dropbox.com/s/0h7g5cjyvxywkt1/deeplabv3_xception_2018_01_04.tar.gz?dl=0 --output-document=./Data/deeplab/v3+/pre-trained_model/deeplabv3_xception_2018_01_04.tar.gz

Download a script which may be used in some of the codes, and make it executable.

    mkdir -p ./bin
    wget https://www.dropbox.com/s/6sdwu3tx9jwzfsm/cp_shapefile?dl=0 --output-document=./bin/cp_shapefile
    chmod a+x ./bin/cp_shapefile

Also also set env as below in the .bashrc:
    
     export PATH=$HOME/bin:$PATH

Lastly, 

    source .bashrc

Clone codes from GitHub:

    git clone https://github.com/yghlc/DeeplabforRS.git ./codes/PycharmProjects/DeeplabforRS
    git clone https://github.com/yghlc/Landuse_DL ./codes/PycharmProjects/Landuse_DL
    git clone https://github.com/yghlc/models.git ./codes/PycharmProjects/tensorflow/yghlc_tf_model


Install python using miniconda 

    wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
    sh Miniconda2-latest-Linux-x86_64.sh -p $HOME/programs/miniconda2 -b

    
Install tensorflow 1.14 (a relative old version) and other python packages. <!-- The installation will run inside 
the container, so we need to submit a job for running singularity. -->
    
    ${HOME}/programs/miniconda2/bin/pip install tensorflow-gpu==1.14
    ${HOME}/programs/miniconda2/bin/conda install gdal=2.3
    ${HOME}/programs/miniconda2/bin/pip install rasterio
    ${HOME}/programs/miniconda2/bin/pip install pyshp==1.2.12
    ${HOME}/programs/miniconda2/bin/pip install rasterstats
    ${HOME}/programs/miniconda2/bin/pip install pillow
    ${HOME}/programs/miniconda2/bin/pip install imgaug==0.2.6
    ${HOME}/programs/miniconda2/bin/pip install geopandas
    ${HOME}/programs/miniconda2/bin/pip install opencv-python==3.4.6.27 (choose a earlier verion to avoid error)
    ${HOME}/programs/miniconda2/bin/pip install GPUtil


## Run training, prediction, and post-processing
After the environment for running Landuse_DL is ready, you can start training your model as well as prediction. 
Suppose your working folder is *test_deeplabV3+_1*, in this folder, a list of files should be presented:
    
    exe.sh
    para.ini
    inf_image_list.txt 
    multi_training_files.txt
    multi_validate_shapefile.txt

*exe.sh* is a script for running the whole process, including preparing training images, 
training, prediction, and post-processing. You may want to comment out some of the lines in this file 
if you just want to run a few steps of them. This file can be copied from *Landuse_DL/thawslumpScripts/exe_qtp.sh*.
In this file, you may also need to modify some input files, such as change *para_qtp.ini* to *para.ini*, 
comment out lines related to *PATH* and *CUDA_VISIBLE_DEVICES*. <!--, and the value of *gpu_num*. -->


*para.ini* is the file where you define input files and parameters. Please edit it accordingly. 
An example of *para.ini* is available at *Landuse_DL/thawslumpScripts/para_qtp.ini*.

*inf_image_list.txt* stores the image file names for prediction as follows,

    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000000000_8bit_Albers.tif
    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000026880_8bit_Albers.tif
    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000053760_8bit_Albers.tif
    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000080640_8bit_Albers.tif
    qtb_sentinel2_2018_JJA_mosaic-0000000000-0000107520_8bit_Albers.tif



To start a job, run 

    ./exe.sh







