#!/bin/bash

# Plot the precision-recall curve
# run in a folder contain results, such as "result_backup"

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 11 February, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

para_file=para.ini
deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py



# plot Precision-recall curve
# get the first ini file,  others should have the same validate polgons
para_file=$(ls BLH_basin_deeplabV3+_1_exp9_iter30000_*imgAug*.ini | head -1)

output=p_r_img_augmentation.jpg


shp_list=$(ls BLH_basin_deeplabV3+_1_exp9_iter30000_post*imgAug*.shp)
python ${deeplabRS}/plot_accuracies.py -p ${para_file} ${shp_list} -o ${output}
