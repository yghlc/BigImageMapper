# Landuse_DL
Land use classification or landform delineation from remote sensing images using Deep Learning. 
This repo contains codes for  mapping thermokarst landforms including thermo-erosion gullies and retrogressive thaw slumps.

We are actively cleaning, re-organizing, and documenting our codes. Please check out the [Release](https://github.com/yghlc/Landuse_DL/tree/release) Branch.

## Citation
If codes here are useful for your project, please cite our papers:

```
@article{huang2019using,
  title={Using Deep Learning to Map Retrogressive Thaw Slumps in the Beiluhe Region (Tibetan Plateau) from CubeSat Images},
  author={Huang, Lingcao and Luo, Jing and Lin, Zhanju and Niu, Fujun and Liu, Lin},
  journal={Remote Sensing of Environment},
  volume = {237},
  year = {2020},
  publisher={ELSEVIER},
  doi = {https://doi.org/10.1016/j.rse.2019.111534}
}
```
```
@article{huang2018automatic,
  title={Automatic Mapping of Thermokarst Landforms from Remote Sensing Images Using Deep Learning: A Case Study in the Northeastern Tibetan Plateau},
  author={Huang, Lingcao and Liu, Lin and Jiang, Liming and Zhang, Tingjun},
  journal={Remote Sensing},
  volume={10},
  number={12},
  pages={2067},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

## How to use
See the script: thawslumpScripts/exe.sh

## Contributions
Please let me know or pull a request if you spot any bug or typo. Thanks!
Any enhancement or new functions are also welcome!

## updates
March 2018:
    Land use classification using the data from 2018_IEEE_GRSS_Data_Fusion. Also submitted the result. codes in "grss_data_fusion". 
    The method utilized Deeplab V4(+3), a semantic segmentation algorithm, to classify land use (20 classes). 
    
August 2018:
    Delineate retrogressive thaw slumps from Planet CubeSat images.
    
January 2019:
    Supporting of Mask RCNN

March 2019:
    Many scripts for producing figures of manuscript: Using Deep Learning to Map Retrogressive Thaw Slumps in the Beiluhe Region (Tibetan Plateau) from CubeSat Images, Remote sensing of Environment, In press

More

## Dependencies and setting:
    Python package: Numpy, rasterio, GDAL 2.3, tensorflow-gpu 1.6, pyshp 1.2.12, pillow, imgaug
    Other: GDAL, OTB, ASP, CUDA 9.0, cudnn 7.0.
    More information on the setting can be found in 'docker_ubuntu1604/run_INsingularity_hlctest.sh' and 'tutorial/Setting_running_on_ITSC.md'
    

## Disclaimer
This is a personal repo that we are actively developing. It may not work as expected. 
We have all the settings on our workstations and servers. You need to spend some efforts on environment settings on your computers before running these codes. 
These codes are only for research, and please take risks by yourself.

## TODO
We will update some of the codes, including bug fix and enhancement because they are used in other projects. 

Better documents
  

