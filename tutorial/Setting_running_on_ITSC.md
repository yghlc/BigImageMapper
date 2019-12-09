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




