#!/bin/bash

eo_dir=/home/hlc/codes/PycharmProjects/Landuse_DL
cd ${eo_dir}
git pull
cd -

output=mosaic_result.tif

# merge patches
### post processing
cd inf_results
   
    python ${eo_dir}/gdal_class_mosaic.py -o ${output} -init 0 *_pred.png
    mv ${output} ../.
cd ..

