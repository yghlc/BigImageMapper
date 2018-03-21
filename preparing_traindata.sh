#!/usr/bin/env bash



para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

eo_dir=/home/hlc/codes/PycharmProjects/DeeplabforRS
root=$(python2 ${para_py} -p ${para_file} working_root)

# current folder (without path)
test_dir=${PWD##*/}

rm -r ${root}/${test_dir}/split_images
rm -r ${root}/${test_dir}/split_labels
mkdir  ${root}/${test_dir}/split_images ${root}/${test_dir}/split_labels

#### preparing training images

### split the training image to many small patch (480*480)
patch_w=$(python2 ${para_py} -p ${para_file} train_patch_width)
patch_h=$(python2 ${para_py} -p ${para_file} train_patch_height)
overlay=$(python2 ${para_py} -p ${para_file} train_pixel_overlay_x)     # the overlay of patch in pixel

trainImg_dir=$(python2 ${para_py} -p ${para_file} input_train_dir)
labelImg_dir=$(python2 ${para_py} -p ${para_file} input_label_dir)

for img in ${trainImg_dir}/*.tif
do
${eo_dir}/split_image.py -W ${patch_w} -H ${patch_h}  -e ${overlay} -o  ${root}/${test_dir}/split_images $img
done
for img in ${labelImg_dir}/*.tif
do
${eo_dir}/split_image.py -W ${patch_w} -H ${patch_h}  -e ${overlay} -o ${root}/${test_dir}/split_labels $img
done


#prepare list files
find ${root}/${test_dir}/split_images/*.tif > list/image_list.txt
find ${root}/${test_dir}/split_labels/*.tif > list/label_list.txt

paste list/image_list.txt list/label_list.txt | awk ' { print $1 " " $2 }' > list/temp.txt
cp list/temp.txt list/train_aug.txt
cp list/image_list.txt list/val.txt
list/extract_fileid.sh list/val


