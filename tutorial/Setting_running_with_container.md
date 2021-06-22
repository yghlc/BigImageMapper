# Setting and running this repo


This file introduces how to set the environment for running this repo.
It also provides a reference for setting and running on your own workstations.


#### Step 1: install miniconda 

Install python using miniconda 

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh -p $HOME/programs/miniconda3 -b

#### Step 2: install tensorflow 1.14 and dependencies for DeepLabv3+

Need: [singularity](https://sylabs.io/singularity) >= 3.6, download a container installed with CUDA 10.0 and cuDNN 7.4, 
which is required by the GPU version of tensorflow 1.14. 

    wget https://www.dropbox.com/s/opxwfc5erdsx8vi/ubuntu2004_cuda1000_cudnn74.sif?dl=0 --output-document=ubuntu2004_cuda1000_cudnn74.sif

Install tensorflow 1.14 (a relative old version) for running [DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab).
    
    wget https://github.com/yghlc/Landuse_DL/blob/master/deeplabBased/tf1.14.yml
    conda env create -f tf1.14.yml

Clone codes from GitHub:

    git clone https://github.com/yghlc/Landuse_DL ./codes/PycharmProjects/Landuse_DL
    git clone https://github.com/yghlc/models.git ./codes/PycharmProjects/tensorflow/yghlc_tf_model
    
    # then set the tensorflow research in the network parameter (e.g.,, deeplabv3plus_xception65.ini):
    tf_research_dir = ~/codes/PycharmProjects/tensorflow/yghlc_tf_model/research
    


## Step 3: Run training, prediction, and post-processing
Activate the conda environment:

    conda activate tf1.14
    # conda deactivate # if want to deactivate tf1.14 environment. 

After the environment for running Landuse_DL is ready, you can start training your model as well as prediction. 
Suppose your working folder is *test_deeplabV3+_1*, in this folder, a list of files should be presented:
    
    runIN_deeplab_sing.sh
    exe.sh
    main_para.ini
    study_area_1.ini
    deeplabv3plus_xception65.ini

*[runIN_deeplab_sing.sh](https://github.com/yghlc/Landuse_DL/blob/master/deeplabBased/runIN_deeplab_sing.sh)* is the script 
to run codes inside the container. 

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

    ./runIN_deeplab_sing.sh

