#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

eo_dir=~/codes/PycharmProjects/Landuse_DL
root=$(python2 ${para_py} -p ${para_file} working_root)

# current folder (without path)
test_dir=${PWD##*/}

if [ -d "${root}/${test_dir}/split_images" ]; then
    rm -r ${root}/${test_dir}/split_images
fi
if [ -d "${root}/${test_dir}/split_labels" ]; then
    rm -r ${root}/${test_dir}/split_labels
fi

mkdir  ${root}/${test_dir}/split_images ${root}/${test_dir}/split_labels

#### preparing training images

### split the training image to many small patch (480*480)
patch_w=$(python2 ${para_py} -p ${para_file} train_patch_width)
patch_h=$(python2 ${para_py} -p ${para_file} train_patch_height)
overlay=$(python2 ${para_py} -p ${para_file} train_pixel_overlay_x)     # the overlay of patch in pixel

trainImg_dir=$(python2 ${para_py} -p ${para_file} input_train_dir)
labelImg_dir=$(python2 ${para_py} -p ${para_file} input_label_dir)

for img in ${trainImg_dir}/*.png
do
${eo_dir}/grss_data_fusion/split_image.py -W ${patch_w} -H ${patch_h}  -e ${overlay} -o  ${root}/${test_dir}/split_images $img
done
for img in ${labelImg_dir}/*.tif
do
${eo_dir}/grss_data_fusion/split_image.py -W ${patch_w} -H ${patch_h}  -e ${overlay} -o ${root}/${test_dir}/split_labels $img
done


#prepare list files
find ${root}/${test_dir}/split_images/*.png > list/image_list.txt
find ${root}/${test_dir}/split_labels/*.png > list/label_list.txt

paste list/image_list.txt list/label_list.txt | awk ' { print $1 " " $2 }' > list/temp.txt
cp list/temp.txt list/train_aug.txt
cp list/image_list.txt list/val.txt
list/extract_fileid.sh list/val

##################################
# rename the label images
output_txt=trainval.txt
if [ ! -f $para_file ]; then
    rm $output_txt
fi


while IFS= read -r line
do
#show the line
echo $line

#split the image and the label path
path=($line)
image_path=${path[0]}
label_path=${path[1]}

#echo $image_path
#echo $label_path

#get the new name of the label
#DIR=$(dirname "${input}")
filename=$(basename "$image_path")
filename_no_ext="${filename%.*}"
#extension="${filename##*.}"

DIR=$(dirname "${label_path}")

mv $label_path $DIR/${filename}

#mv corresponding xml file
mv ${label_path}.aux.xml $DIR/${filename}.aux.xml

#output file name without extension
echo $filename_no_ext >> $output_txt

done < "list/train_aug.txt"

mkdir list/old_txt
mv list/*.txt list/old_txt/.
mv $output_txt list/.

# copy the training data for elevation
cp list/$output_txt list/val.txt
