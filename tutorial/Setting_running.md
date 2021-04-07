# Setting and running this repo


This file introduces how to set the environment for running this repo.
It also provides a reference for setting and running on your own workstations.


#### Step 1: install packages and dependencies

Install python using miniconda 

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh -p $HOME/programs/miniconda3 -b

Install tensorflow 1.14 (a relative old version) for running [DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab)
    
    conda create -n tf1.14 python=3.7   # need python3.7 or 3.6 (python 3.8 cannot found 1.x version)
    conda (or source) activate tf1.14  # after this step, it will show "tf1.14" at the begin of the line
    # check where is pip by "which pip", making sure it is under the environment of tf1.14
    pip install tensorflow-gpu==1.14   # for GPU version  or 
    #pip install tensorflow==1.14       # for CPU version
    pip install tf_slim==1.0.0       # 1.0.0 is required by mobilenet_v3
    pip install numpy==1.16.4       # use a relative old version to avoid warning.
    pip install gast==0.2.2         # use a relative old version to avoid warning.
    pip install pillow
    pip install opencv-python==3.4.6.27 (choose a earlier verion to avoid error)
    pip install conda
    conda install gdal=2.4.2
    pip install rasterio
    
    which python  # output the path of python then set tf1x_python in network parameter (e.g., deeplabv3plus_xception65.ini):
    tf1x_python  = ~/programs/anaconda3/envs/tf1.14/bin/python 

Install other python packages under tf.14 (suggested) or default python. <!-- The installation will run inside 
the container, so we need to submit a job for running singularity. -->
    
    pip install pyshp==1.2.12
    pip install rasterstats
    pip install imgaug==0.2.6
    pip install geopandas
    pip install GPUtil


If we run the GPU version of tensorflow 1.14, we need to install CUDA and cuDNN (on Ubuntu). 

Install CUDA 10.0 and cuDNN 7.4, these are required by tensorflow 1.14. \
Ubuntu 18 or 20 already may have CUDA 10.0 or 10.2 installed.
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

Clone codes from GitHub:

    git clone https://github.com/yghlc/Landuse_DL ./codes/PycharmProjects/Landuse_DL
    git clone https://github.com/yghlc/models.git ./codes/PycharmProjects/tensorflow/yghlc_tf_model
    
    # then set the tensorflow research in the network parameter (e.g.,, deeplabv3plus_xception65.ini):
    tf_research_dir = ~/codes/PycharmProjects/tensorflow/yghlc_tf_model/research
    


## Run training, prediction, and post-processing
After the environment for running Landuse_DL is ready, you can start training your model as well as prediction. 
Suppose your working folder is *test_deeplabV3+_1*, in this folder, a list of files should be presented:
    
    exe.sh
    main_para.ini
    study_area_1.ini
    deeplabv3plus_xception65.ini

*exe.sh* is a script for running the whole process, including preparing training images, 
training, prediction, and post-processing. You may want to comment out some of the lines in this file 
if you just want to run a few steps of them. This file can be copied from *Landuse_DL/working_dir/exe.sh*.
In this file, you may also need to modify some input files, such as change *para_qtp.ini* to *para.ini*, 
comment out lines related to *PATH* and *CUDA_VISIBLE_DEVICES*. <!--, and the value of *gpu_num*. -->


*main_para.ini* is the file where you define main parameters. Please edit it accordingly. 
An example of *main_para.ini* is available at *Landuse_DL/working_dir/main_para.ini*.

*study_area_1.ini* is a file to define a study region, including training polygons, images, 
and elevation files (if available). Please edit it accordingly. We can have multiple region-defined files. 

*deeplabv3plus_xception65.ini* is the file to define the deep learning network and some training parameters. 
Please edit it accordingly.



To start a job, run 

    ./exe.sh







