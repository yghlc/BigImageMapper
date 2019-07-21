#!/usr/bin/env bash

#introduction: Run prediction using trained model of DL
# copy this file to working directory, then modify it accordingly.
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 20 July, 2019


#MAKE SURE the /usr/bin/python, which is python2 on Cryo06
export PATH=/usr/bin:$PATH
# python2 on Cryo03, tensorflow 1.6
export PATH=~/programs/anaconda2/bin:$PATH

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

eo_dir=~/codes/PycharmProjects/Landuse_DL
cd ${eo_dir}
git pull
cd -

## modify according to test requirement or environment (only used on GPU for prediction)
#set GPU on Cryo06
export CUDA_VISIBLE_DEVICES=1
#set GPU on Cryo03
export CUDA_VISIBLE_DEVICES=0
gpu_num=1
para_file=para.ini
para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py
train_dir=Data/Qinghai-Tibet/beiluhe/beiluhe_sentinel-2/autoMapping/BLH_deeplabV3+_1

inf_dir=inf_results

################################################
SECONDS=0
# remove previous data or results if necessary
rm -r inf_results | true
rm processLog.txt

################################################
#copy trained model to here

expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
mkdir -p ${expr_name}/export

NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
trail=iter${NUM_ITERATIONS}
frozen_graph=frozen_inference_graph_${trail}.pb


scp $chpc_host:~/${train_dir}/${expr_name}/export/${frozen_graph}  ${expr_name}/export/${frozen_graph}

SECONDS=0
################################################
## inference and post processing, including output "time_cost.txt"
#${eo_dir}/thawslumpScripts/inf.sh ${para_file}
inf_batch_size=$(python2 ${para_py} -p ${para_file} inf_batch_size)
python ${eo_dir}/grss_data_fusion/deeplab_inference.py --frozen_graph=${frozen_graph} --inf_output_dir=${inf_dir} --inf_batch_size=${inf_batch_size}


duration=$SECONDS
echo "$(date): time cost of inference: ${duration} seconds">>"time_cost.txt"

################################################
# post processing and backup results
${eo_dir}/sentinelScripts/postProc_qtp.sh ${para_file} s2_qtb_exp1
