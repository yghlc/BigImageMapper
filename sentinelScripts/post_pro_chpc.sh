#!/bin/bash


#introduction: copy inference patches from chpc cluster, then post them in the local machine
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 13 October, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

para_file=para_qtp.ini
eo_dir=~/codes/PycharmProjects/Landuse_DL

script=~/codes/PycharmProjects/Landuse_DL/sentinelScripts/copyTolocal_postPro.py

#${script} para_qtp.ini
# parallel run this, delay 200 second for each job
cmd="${script} para_qtp.ini"
#parallel --progress --delay 200 ${script} ${script} ${script} ${script} ${script} ${script} ${script} ${script} ::: para_qtp.ini para_qtp.ini para_qtp.ini para_qtp.ini para_qtp.ini para_qtp.ini para_qtp.ini para_qtp.ini
parallel --progress --delay 200 ${cmd} ${cmd} ::: 1 2

## post processing and copy results, including output "time_cost.txt"
test_name=chpc_1
${eo_dir}/sentinelScripts/postProc_qtp.sh ${para_file}  ${test_name}
## merge polygons
${eo_dir}/sentinelScripts/merge_shapefiles.sh ${para_file} ${test_name}
