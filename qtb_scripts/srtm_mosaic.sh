#!/bin/bash

#introduction: get SRTM mosaic, reproject, and crop the QTP extent
# run this script in ~/Data/Qinghai-Tibet/qtp_dem

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 1 August, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

SECONDS=0


output=qtp_srtm_1arc_v3.tif
output_30m=qtp_srtm_v3_30m.tif
output_30m_crop=qtp_srtm_v3_30m_crop.tif
res=30

tif_dir="Bulk Order 1026517/SRTM 1 Arc-Second Global"

# merge SRTM patches patches
if [ ! -f ${output} ]; then
    gdal_merge.py -o ${output} ${tif_dir}/*.tif
else
    echo "${output} already exist, skip gdal_merge.py"
fi


# reproject the shapefile from "GEOGCS (WGS84)" to "Cartesian (XY) projection"
# the following projection (wkt string) came from ran.shp (gdalsrsinfo -o wkt ran.shp), the Permafrost map on the Tibetan Plateau
# need to modify it if switch to other regions
t_srs="PROJCS["Krasovsky_1940_Albers",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_84",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["longitude_of_center",90.0],PARAMETER["Standard_Parallel_1",27.5],PARAMETER["Standard_Parallel_2",37.5],PARAMETER["latitude_of_center",0.0],UNIT["Meter",1.0]]"

# merge SRTM patches patches
if [ ! -f ${output_30m} ]; then
    gdalwarp -overwrite -s_srs EPSG:4326 -t_srs ${t_srs} -tr ${res} ${res} \
    -of GTiff ${output} ${output_30m}
else
    echo "${output_30m} already exist, skip reproject"
fi

# crop the DEM
extent=~/Data/Qinghai-Tibet/Qinghai-Tibet_Plateau_shp/Qinghai-Tibet_Plateau_outline2_gee/QTP_outline_simplified_0.1.shp
if [ ! -f ${output_30m_crop} ]; then
    gdalwarp -cutline ${extent} -crop_to_cutline -tr ${res} ${res} \
    -of GTiff ${output_30m}.tif ${output_30m_crop}
else
    echo "${output_30m_crop} already exist, skip cropping"
fi


duration=$SECONDS
echo "$(date): time cost of creating QTP-SRTM mosaic: ${duration} seconds">>"time_cost.txt"
