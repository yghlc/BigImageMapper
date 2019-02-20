#!/bin/bash

# output the table of accuracies
# run in a folder contain results, such as "result_backup"

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 19 February, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

para_file=para.ini
deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py

script=~/codes/PycharmProjects/Landuse_DL/resultScript

# plot Precision-recall curve
# get the first ini file,  others should have the same validate polgons

acc_log=accuracies_log.txt
ap=$(ls *_ap.txt)

fold_name=$(basename $PWD)

output=fold_name_accuracy_table.csv

python ${script}  ${acc_log} ${ap} time_cost.txt -o ${output}
