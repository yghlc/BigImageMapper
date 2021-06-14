#!/usr/bin/env bash

# bash create new region defined parameter files (ini)

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 13 June, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

py=~/codes/PycharmProjects/Landuse_DL/utility/create_region_ini.py


function new_ini(){
    dir=$1
    template_ini=$2
    name=$3
    ${py} ${dir} ${template_ini} -n ${name} -i
}

rm region_ini_files.txt || true
rm *.ini || true

ref_ini=~/Data/flooding_area/automapping/houston_deeplabV3+_1/area_Houston.ini

region=Houston
dir=~/Bhaltos2/lingcaoHuang/flooding_area/Houston/Houston_SAR_GRD_FLOAT_gee/S1_Houston_prj_8bit_select
new_ini ${dir} ${ref_ini} ${region} 


region=Goalpara
dir=~/Bhaltos2/lingcaoHuang/flooding_area/Goalpara/Goalpara_power_transform_prj_8bit
new_ini ${dir} ${ref_ini} ${region}


region=Vadodara_lower
dir=~/Bhaltos2/lingcaoHuang/flooding_area/Vadodara/Vadodara_2020_lower_power_transform_prj_8bit
new_ini ${dir} ${ref_ini} ${region}


region=Vadodara_upper
dir=~/Bhaltos2/lingcaoHuang/flooding_area/Vadodara/Vadodara_2020_upper_power_transform_prj_8bit
new_ini ${dir} ${ref_ini} ${region}






