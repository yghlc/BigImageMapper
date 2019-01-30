#!/bin/bash

# orthorectify using mapproject (aps)
# run this script in the SPOT 5 folder: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_spot

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH
#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-Linux/bin:$PATH
export PATH=~/programs/StereoPipeline-2.6.1-2019-01-19-x86_64-Linux/bin:$PATH

#number of thread of to use, 8 or 16 on linux, 4 on mac (default is 4)
num_thr=16

#dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30.tif
dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30_prj_utm.tif

out_res=2.5
nodata=0

outdir=spot5_orthorectified
mkdir -p ${outdir}

#test_roi="--t_pixelwin 0 0 500 500"
test_roi=

function ortho_rectify() {
    local spot_tif=$1
    local spot_dim=$2

    SECONDS=0

    datasetName=$(cat ${spot_dim} | grep DATASET_NAME)
    IFS=' ' read -r -a array <<< "$datasetName"
    date=${array[3]}
    date=$(echo 20$date | tr / - )

    pathrow=${array[2]}

    color=${array[6]}
    IFS='<' read -r -a array <<< "$color"
    color=${array[0]}
    color=$(echo $color | tr + - )

    output=beilue_spot5_${color}_${date}_${pathrow}.tif

    echo start ortho rectification: $output

    # example
#    mapproject sample_dem.tif front/SEGMT01/IMAGERY.BIL front/SEGMT01/METADATA.DIM
#      front_map_proj.tif -t rpc

    add_spot_rpc ${spot_dim} -o ${spot_dim}
#    exit 0
    # ortho
    mapproject -t rpc --nodata-value ${nodata} --tr ${out_res} ${dem} ${spot_tif} ${spot_dim} ${output} \
        ${test_roi} --threads ${num_thr} --ot Byte --tif-compress None

    # mv results
    mv  ${output}  ${outdir}/.

    #exit
    duration=$SECONDS
    echo "$(date): time cost of orthorectification of ${output}: ${duration} seconds">>"time_cost.txt"
}


for spot5 in $(ls SWH*/SCEN*/*.TIF); do

    echo $spot5 >> "time_cost.txt"
    dir=$(dirname $spot5)
	filename=$(basename -- "$spot5")
#	extension="${filename##*.}"
	filename_no_ext="${filename%.*}"

    dimfile=${dir}/METADATA.DIM

    ortho_rectify $spot5  $dimfile
done





