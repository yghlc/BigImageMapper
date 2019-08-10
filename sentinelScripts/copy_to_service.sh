#!/usr/bin/env bash

# due to the home folder limit (56G) on ITSC service, we cannot scp all files (around 160 to service) in one time.
# because, ever I copied to /project folder, the group info is still s1155090023, not LinLiu.
# so I separate files, then copied to ITSC service, then change the group to LinLiu (so it will not use the storage of home folder)

# total 78 tif files.


# copy files
function copy_files() {
    file_head=$1

    scp $1 $chpc_host:/project/LinLiu/hlc/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2/8bit_dir/sentinel-2_2018_mosaic_v3/.
}


copy_files qtb_sentinel2_2018_mosaic-0000000000*
# stop, then change the group info on ITSC services

#copy_files qtb_sentinel2_2018_mosaic-0000026880*

#copy_files qtb_sentinel2_2018_mosaic-0000053760*

#copy_files qtb_sentinel2_2018_mosaic-0000080640*

#copy_files qtb_sentinel2_2018_mosaic-0000107520*

#copy_files qtb_sentinel2_2018_mosaic-0000134400*


# copy txt and sh files (no need)

copy_files *.txt
copy_files *.sh





