#!/bin/bash

#upcodes.sh

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

para_file=para_qtp.ini
eo_dir=~/codes/PycharmProjects/Landuse_DL

script=~/codes/PycharmProjects/Landuse_DL/sentinelScripts/copyTolocal_postPro.py

#~/codes/PycharmProjects/Landuse_DL/sentinelScripts/copyTolocal_postPro.py para_qtp.ini
# parallel run this, delay 180 second for each job
cmd="~/codes/PycharmProjects/Landuse_DL/sentinelScripts/copyTolocal_postPro.py para_qtp.ini"
parallel --progress --delay 180 ${cmd} ${cmd} ${cmd} ::: para_qtp.ini para_qtp.ini para_qtp.ini 


## post processing and copy results, including output "time_cost.txt"
test_name=chpc_1
${eo_dir}/sentinelScripts/postProc_qtp.sh ${para_file}  ${test_name}
## merge polygons
${eo_dir}/sentinelScripts/merge_shapefiles.sh ${para_file} ${test_name}
