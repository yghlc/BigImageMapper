#!/bin/bash

# calculate slope and aspect using DEM
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

slope=${outdir}/beiluhe_srtm30_utm_basinExt_sagaSlope
aspect=${outdir}/beiluhe_srtm30_utm_basinExt_sagaAspect

# import raster (convert to tif to grid format)
dem_grid=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_srtm30_utm_basinExt.sdat
if [ ! -f $dem_grid ]; then
    #
    run_exe_indocker "saga_cmd io_gdal 0  -GRIDS ${dem_grid} -FILES ${dem}"
else
    echo "warning: $dem_grid already exist "
fi

#run_exe_indocker "gdalinfo ${dem} "

#
#run_exe_indocker "saga_cmd ta_morphometry 0"
# method 2: Method: least squares fitted plane, output the same result as QGIS
# method 3: Method: 6 parameter 2nd order polynom (Evans 1979), output the same result as QGIS
# method 4: Method: 6 parameter 2nd order polynom (Heerdegen & Beran 1982), output the same result as QGIS
# method 5: Method: 6 parameter 2nd order polynom (Bauer, Rohdenburg, Bork 1985), output the same result as QGIS
# method 6: Method: 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987), output is slightly different with QGIS
# method 7: 10 parameter 3rd order polynom (Haralick 1983), output is slightly different with QGIS
run_exe_indocker "saga_cmd ta_morphometry 0 -ELEVATION ${dem_grid} -SLOPE ${slope} -ASPECT ${aspect} \
 -UNIT_SLOPE 1 -UNIT_ASPECT 1 -METHOD 5"

# to tif format
run_exe_indocker "saga_cmd io_gdal 1 -GRIDS ${slope}.sdat -FILE ${slope}.tif -FORMAT 1 "
run_exe_indocker "saga_cmd io_gdal 1 -GRIDS ${aspect}.sdat -FILE ${aspect}.tif -FORMAT 1 "




