#!/bin/bash

para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

eo_dir=/home/hlc/codes/PycharmProjects/Landuse_DL

#eo_dir=$(python2 ${para_py} -p ${para_file} codes_dir)
augscript=${eo_dir}/grss_data_fusion/image_augment.py

SECONDS=0

# Helper function to update the list
function update_listfile() {
    for png in $(ls *.png)
    do
        filename=$(basename "$png")
        filename_no_ext="${filename%.*}"
        echo $filename_no_ext >> "trainval.txt"
    done
    mv "trainval.txt" ../.
}

#augment training images
cd split_images
${augscript} -d ./ -e .png ../list/trainval.txt -o ./

update_listfile
cd ..

#augment training lables
cd split_labels
${augscript} -d ./ -e .png --is_ground_truth ../list/trainval.txt -o ./

# have same list, so we don't need to update again
#update_listfile
# replace the 0 pixel as 255
for png in $(ls *_R*.png); do
    ${eo_dir}/remove_zero_pixels.py $png temp.png
    mv temp.png $png
done


cd ..

# copy the training data for elevation
mv trainval.txt list/.
cp list/trainval.txt list/val.txt

duration=$SECONDS
echo "$(date): time cost of preparing training images augmentation: ${duration} seconds">>"time_cost.txt"