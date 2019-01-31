#!/bin/bash

# orthorectify using mapproject (aps)
# run this script in the ZY3 image folder contaning NAD images: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3/ZY3_NAD_E92.8_N35.0_20141207_L1A0002929919

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH
#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-Linux/bin:$PATH
export PATH=~/programs/StereoPipeline-2.6.1-2019-01-19-x86_64-Linux/bin:$PATH

#number of thread of to use, 8 or 16 on linux, 4 on mac (default is 4)
num_thr=16

#dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30.tif
dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30_prj_utm.tif

outdir=zy3_nad_ortho

mkdir -p ${outdir}

# function of converting to 8bit using gdal_translate with max and min value.
function ortho_rectify() {
    local folder=$1
    local out_res=$2


    SECONDS=0

    cd ${folder}

    str=$(basename $PWD)
    IFS='_ ' read -r -a array <<< "$str"
    output=${array[0]}_${array[4]}_${array[5]}_prj.tif

    tifname=$(ls *-NAD.tiff)
    #filename_no_ext
    prename="${tifname%.*}"

#     ${prename}.xml
#     sicne ASP complain the xml is not recognised, then remove it. the script still work without this complaint
#     on Linux, no METADATATYPE, we set METADATATYPE=ZY3,
#     on Mac, it set METADATATYPE=ZY3, and can not be modified, and mapproject copy the rpb files
#     Same version, but different behaviour in Mac and Linux, strange
    mapproject -t rpc $dem ${prename}.tiff  ${output} --mpp ${out_res} \
        --ot UInt16 --tif-compress None --mo METADATATYPE=ZY3 --threads ${num_thr}

    cd -


    # mv results
    mv ${folder}/${output}  ${outdir}/.

    #exit
    duration=$SECONDS
    echo "$(date): time cost of orthorectification of ${folder}: ${duration} seconds">>"time_cost.txt"
}


for zy3_nad in $(ls -d ZY3_NAD* |grep -v gz); do
    echo $zy3_nad

    ortho_rectify $zy3_nad 2.1
done

for zy302_tms in $(ls -d ZY302_TMS* |grep -v gz ); do
    echo $zy302_tms

    ortho_rectify $zy302_tms 2.1
done