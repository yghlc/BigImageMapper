#!/bin/bash

# extract dem from ZY3 FWD and BWD images
# run this script in the ZY3 image folder contaning NAD images: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3/ZY3_DLC_E92.8_N35.0_20141207_L1A0002929919

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH
#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-Linux/bin:$PATH
export PATH=~/programs/StereoPipeline-2.6.1-2019-01-19-x86_64-Linux/bin:$PATH

#number of thread of to use, 8 or 16 on linux, 4 on mac (default is 4)
num_thr=16


#dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30.tif
dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30_prj_utm.tif

#NADname=$(ls *-NAD.tiff)
#filename_no_ext
#prename="${tifname%.*}"
FWDname=$(ls *-FWD.tiff)
BWDname=$(ls *-BWD.tiff)

# one solution: use NAD as the left image, FWD or BWD as the right image?
left=${FWDname}
right=${BWDname}

# input image resolution zy3: 3.5m zy3-02: 2.5m
#out_res=3.5
out_res=$1
# output dem resolution, can set a higher value for zy3-02
dem_res=3.0

str=$(basename $PWD)
IFS='_ ' read -r -a array <<< "$str"
output=stereo/zy3_${array[4]}_${array[5]}

# for test: xmin ymin xmax ymax
#test_roi="--t_pixelwin 5000 5000 6000 6000"
test_roi=

# Running Stereo with Map-projected Images
# In this mode, ASP does not create a terrain model from scratch,
# but rather uses an existing terrain model as an initial guess, and improves on it.

# notice that, space or other character after '\' will cause trouble. This could happen when copy and paste

mapproject -t rpc --tr ${out_res} ${dem} ${left} left_mapped.tif \
        ${test_roi} --threads ${num_thr} --ot UInt16 --tif-compress None

mapproject -t rpc --tr ${out_res} ${dem} ${right} right_mapped.tif \
        ${test_roi} --threads ${num_thr} --ot UInt16 --tif-compress None

# copy rpc file on Linux
cp "${left%.*}".rpb left_mapped.RPB
cp "${right%.*}".rpb right_mapped.RPB

stereo -t rpcmaprpc --subpixel-mode 3 --alignment-method none     \
           left_mapped.tif right_mapped.tif --threads ${num_thr}    \
           ${output} ${dem}


#parallel_stereo -t rpc --stereo-algorithm 2 --subpixel-mode 3 --alignment-method none \
#    ${left} ${right} --threads ${num_thr}  ${output}

# create dem from the point cloud (PC) file, --search-radius-factor 5 or higher to fill holes
# --search-radius-factor 10
# output two resolution, one is the same to input, the other is user defined
# the output log: "Percentage of valid pixels = 0.979167" shows that there are a few pixels with nodata (holes and edge pixels)
#point2dem  --search-radius-factor 10 --nodata-value -9999 --tr "${out_res} ${dem_res}"  ${output}-PC.tif
point2dem  --search-radius-factor 10 --nodata-value -9999 --tr ${dem_res} ${output}-PC.tif \
    --threads ${num_thr} --tif-compress None


# post-processing: fill holes
# If the resulting DEM turns out to be noisy or have holes,
# one could change in point2dem the search radius factor, use hole-􏰟filling,
# invoke more aggressive outlier removal, and erode pixels at the boundary
# (those tend to be less reliable). Alternatively, holes can be filled with dem_mosaic.


# Creating DEMs Relative to the Geoid/Areoid, after this the dem is close the SRTM (difference < a few meters)
dem_geoid  ${output}-DEM.tif --tif-compress None --threads ${num_thr}