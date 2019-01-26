#!/bin/bash

# create dsm from each zy3 scene
# run this script in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 25 January, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

outdir=zy3_dsm_files

mkdir -p ${outdir}

# function of converting to 8bit using gdal_translate with max and min value.
function extract_dsm() {
    local folder=${1}
    local res=${2}

    cd ${folder}

    SECONDS=0
    # extract dsm
    cp ~/codes/PycharmProjects/Landuse_DL/zy3Scripts/stereo.default .
    ~/codes/PycharmProjects/Landuse_DL/zy3Scripts/extract_dem.sh ${res}

    cd -


    # mv results
    mv ${folder}/stereo/*-DEM.tif  ${outdir}/.
    mv ${folder}/*-DEM-adj.tif ${outdir}/.

    # remove folder to save storage
    rm -r ${folder}/stereo

    #exit
    duration=$SECONDS
    echo "$(date): time cost of extracting dsm of ${folder}: ${duration} seconds">>"time_cost.txt"
}

for zy3_dlc in $(ls -d ZY3_DLC* |grep -v gz); do
    echo $zy3_dlc

    extract_dsm $zy3_dlc 3.5
done

for zy302_tms in $(ls -d ZY302_TMS* |grep -v gz ); do
    echo $zy302_tms

    extract_dsm $zy302_tms 2.5
done






