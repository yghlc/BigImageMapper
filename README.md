# Landuse_DL
Land use classification or landform delineation from remote sensing images using Deep Learning. This repo contains codes for  mapping thermokarst landforms including thermo-ersion gullies and retrogressive thaw slumps.

## Citation
If any code here is useful for your project, please consider citing our papers:

```
@article{huang2019using,
  title={Using Deep Learning to Map Retrogressive Thaw Slumps in the Beiluhe Region (Tibetan Plateau) from CubeSat Images},
  author={Huang, Lingcao and Luo, Jing and Lin, Zhanju and Niu, Fujun and Liu, Lin},
  journal={Remote Sensing of Environment},
  In press
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
See "tutorial" . 

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


## Dependencies:
    Python package: Numpy, sklearn, rasterio, GDAL. Binary package: GDAL, OTB, ASP 

## Disclaimer
This is a personal repo that we are actively developing. It may not work as expected. 

## TODO
We will update some of the codes, including bug fix and enhancement because they are used in other projects. 
If many requests (at least 10) to repeat our results, we will provide a docker distribution and demo. 

Better documents
  

