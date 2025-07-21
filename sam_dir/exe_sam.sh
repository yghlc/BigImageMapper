#!/usr/bin/env bash

#introduction: Run the whole process of segment landforms from remote sensing images using Segment Anything model
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 16 June, 2023

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


# set GPUs want to used (e.g. use GPU 0 & 1)
#export CUDA_VISIBLE_DEVICES=0,1

# the main parameter files
para_file=main_para.ini

# BigImageMapper scripts dir
eo_dir=~/codes/PycharmProjects/BigImageMapper

################################################
SECONDS=0
# remove previous data or results if necessary
rm time_cost.txt || true

# prepare prompts (points or boxes)
${eo_dir}/sam_dir/trainingPolys_to_prompts.py ${para_file}

duration=$SECONDS
echo "$(date): time cost of preparing data points: ${duration} seconds">>"time_cost.txt"

################################################
## run within conda environment (name: pytorch)
### segment
rm -r multi_segment_results || true
#conda run --no-capture-output -n pytorch bash -c "${eo_dir}/sam_dir/sam_predict.py ${para_file}"
${eo_dir}/sam_dir/sam_predict.py ${para_file}
################################################


## post processing and copy results, inf_post_note indicate notes for inference and post-processing
inf_post_note=1
${eo_dir}/workflow/postProcess.py ${para_file}  ${inf_post_note}
