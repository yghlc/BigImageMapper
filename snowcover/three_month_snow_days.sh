#!/bin/bash

# get three month snow cover days from monthly data

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 29 May, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace



dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt


outdir=${dir}/JanFebMar2003to2012
#Jan, Feb, Mar
#mkdir -p ${outdir}
#rm ${outdir}/* || true
#for year in $(seq 2003 2012); do
#
#    a_file=$(ls ${dir}/*2003to2012/snow_days_${year}_1.tif)
#    b_file=$(ls ${dir}/*2003to2012/snow_days_${year}_2.tif)
#    c_file=$(ls ${dir}/*2003to2012/snow_days_${year}_3.tif)
#    echo $year
#
#    echo $a_file
#
#    output=snow_days_${year}_1_2_3.tif
#
#    gdal_calc.py -A ${a_file} -B ${b_file} -C  ${c_file} --outfile=${outdir}/${output} --calc="A+B+C"
#
##    exit
#done



outdir=${dir}/AprMayJun2003to2012
#Apr, May, Jun
mkdir -p ${outdir}
rm ${outdir}/* || true
for year in $(seq 2003 2012); do

    a_file=$(ls ${dir}/*2003to2012/snow_days_${year}_4.tif)
    b_file=$(ls ${dir}/*2003to2012/snow_days_${year}_5.tif)
    c_file=$(ls ${dir}/*2003to2012/snow_days_${year}_6.tif)
    echo $year

    echo $a_file

    output=snow_days_${year}_4_5_6.tif

    gdal_calc.py -A ${a_file} -B ${b_file} -C  ${c_file} --outfile=${outdir}/${output} --calc="A+B+C"

#    exit
done



#Jul, Aug, Sep
outdir=${dir}/JulAugSep2003to2012
mkdir -p ${outdir}
rm ${outdir}/* || true
for year in $(seq 2003 2012); do

    a_file=$(ls ${dir}/*2003to2012/snow_days_${year}_7.tif)
    b_file=$(ls ${dir}/*2003to2012/snow_days_${year}_8.tif)
    c_file=$(ls ${dir}/*2003to2012/snow_days_${year}_9.tif)
    echo $year

    echo $a_file

    output=snow_days_${year}_7_8_9.tif

    gdal_calc.py -A ${a_file} -B ${b_file} -C  ${c_file} --outfile=${outdir}/${output} --calc="A+B+C"

#    exit
done


#Oct, Nov, Dec
outdir=${dir}/OctNovDec2003to2012
mkdir -p ${outdir}
rm ${outdir}/* || true
for year in $(seq 2003 2012); do

    a_file=$(ls ${dir}/*2003to2012/snow_days_${year}_10.tif)
    b_file=$(ls ${dir}/*2003to2012/snow_days_${year}_11.tif)
    c_file=$(ls ${dir}/*2003to2012/snow_days_${year}_12.tif)
    echo $year

    echo $a_file

    output=snow_days_${year}_10_11_12.tif

    gdal_calc.py -A ${a_file} -B ${b_file} -C  ${c_file} --outfile=${outdir}/${output} --calc="A+B+C"

#    exit
done
