#!/bin/bash

# calculate Potential Incoming Solar Radiation based on DEM (PISR)
# run this script in ~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30
# since this is run inside docker, requires all the input and output are absolute path

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 8 February, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

outdir=${PWD}/dem_derived

mkdir -p ${outdir}

# get container id (-q: only get the id, -l: get the latest one)
container_id=$(docker ps --filter "ancestor=saga_gis:latest" -q -l)

# run command in the container
function run_exe_indocker() {
    local cmd=${1}

    # start the docker contain and mount the data folder (include the soft link and contain all the data):
    # on Mac (don't have the issue of file permsssion)
    # docker run -d -v $HOME/Data:$HOME/Data --user $(whoami) -it  saga_gis
    # on Cryo06, set the user and group id (1000:1000 for hlc on Cryo06, output by: cat /etc/passwd | grep hlc)
    # docker run -d -v $HOME/Data:$HOME/Data --user 1000:1000 -it  saga_gis  # 1000 is the hlc user ID on cryo06
    # if the container exit, then RUN "docker start id" to start it again, or allow it to restart automatically

    docker exec -it ${container_id} ${cmd}

}

# calculate Slope, Aspect, and Curvature

dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_srtm30_utm_basinExt.tif

pisr=${outdir}/beiluhe_srtm30_utm_basinExt_PISR_total
pisr_direct=${outdir}/beiluhe_srtm30_utm_basinExt_PISR_direct
pisr_diffus=${outdir}/beiluhe_srtm30_utm_basinExt_PISR_diffus

pisr_flat=${outdir}/beiluhe_srtm30_utm_basinExt_PISR_flat
pisr_duration=${outdir}/beiluhe_srtm30_utm_basinExt_PISR_duration



# import raster (convert to tif to grid format)
dem_grid=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_srtm30_utm_basinExt.sdat
if [ ! -f $dem_grid ]; then
    #
    run_exe_indocker "saga_cmd io_gdal 0  -GRIDS ${dem_grid} -FILES ${dem}"
else
    echo "warning: $dem_grid already exist "
fi

#run_exe_indocker "gdalinfo ${dem} "


run_exe_indocker "saga_cmd ta_lighting 2 -GRD_DEM ${dem_grid} -GRD_DIRECT ${pisr_direct} -GRD_DIFFUS ${pisr_diffus}  \
-GRD_TOTAL ${pisr} -GRD_FLAT ${pisr_flat} -GRD_DURATION  ${pisr_duration} -LOCATION 1 \
-PERIOD 2 -DAY 2018-06-01 -DAY_STOP 2018-08-31 -DAYS_STEP 1 "

# to tif format
run_exe_indocker "saga_cmd io_gdal 1 -GRIDS ${pisr}.sdat -FILE ${pisr}.tif -FORMAT 1 "
run_exe_indocker "saga_cmd io_gdal 1 -GRIDS ${pisr_direct}.sdat -FILE ${pisr_direct}.tif -FORMAT 1 "
run_exe_indocker "saga_cmd io_gdal 1 -GRIDS ${pisr_diffus}.sdat -FILE ${pisr_diffus}.tif -FORMAT 1 "
run_exe_indocker "saga_cmd io_gdal 1 -GRIDS ${pisr_flat}.sdat -FILE ${pisr_flat}.tif -FORMAT 1 "
#run_exe_indocker "saga_cmd io_gdal 1 -GRIDS ${pisr_duration}.sdat -FILE ${pisr_duration}.tif -FORMAT 1 "





