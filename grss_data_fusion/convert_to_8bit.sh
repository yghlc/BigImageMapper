#!/usr/bin/env bash



# function of converting to 8bit using gdal_translate with max and min value.
function convert_to8bit() {
    local minValue=${1}
    local maxValue=${2}

    local input=${3}

     # get file name without extension
    DIR=$(dirname "${input}")
    filename=$(basename "$input")
    extension="${filename##*.}"
    filename="${filename%.*}"

    output=${DIR}/${filename}_8bit.png

    # use PNG because tensorflow does not support TIF
    gdal_translate -of PNG -ot Byte -scale ${minValue} ${maxValue} 0 255 ${input} ${output}
}

# function of converting to 8bit using gdal_translate with max and min value.
function convert_to16bit() {
    local minValue=${1}
    local maxValue=${2}

    local input=${3}

     # get file name without extension
    DIR=$(dirname "${input}")
    filename=$(basename "$input")
    extension="${filename##*.}"
    filename="${filename%.*}"

    output=${DIR}/${filename}_16bit.png

    # PNG driver doesn't support data type Int16. Only eight bit (Byte) and sixteen bit (UInt16) bands supported. Defaulting to Byte
    gdal_translate -of PNG -ot UInt16 -scale ${minValue} ${maxValue} 0 255 ${input} ${output}
}


# run this in ~/Data/2018_IEEE_GRSS_Data_Fusion/2018_Release_Phase1/Lidar_GeoTiff_Rasters

cd ~/Data/2018_IEEE_GRSS_Data_Fusion/2018IEEE_Contest/Phase2/Lidar_GeoTiff_Rasters

#gdal_translate -ot Byte -scale 210 3845 0 255 UH17_GI1F051_TR.tif UH17_GI1F051_TR_8bit.tif

# the max and min value is copy for the whole datasets of Lidar Intensity

convert_to8bit 210 3845 Intensity_C1/UH17_GI1F051.tif

convert_to8bit 207 5174 Intensity_C2/UH17_GI2F051.tif

convert_to8bit 144 8583 Intensity_C3/UH17_GI3F051.tif



convert_to16bit 210 3845 Intensity_C1/UH17_GI1F051_TR.tif

convert_to16bit 207 5174 Intensity_C2/UH17_GI2F051_TR.tif

convert_to16bit 144 8583 Intensity_C3/UH17_GI3F051_TR.tif







