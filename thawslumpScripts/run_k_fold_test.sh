#!/bin/bash

# run some test

# Exit immediately if a command exits with a non-zero status.
set -e

eo_dir=~/codes/PycharmProjects/Landuse_DL
cd ${eo_dir}
git pull
cd -

#test image augmentation
#${eo_dir}/thawslumpScripts/test_img_augment.sh

# test on k-fold cross validation
${eo_dir}/thawslumpScripts/test_kfold_cross_val_multi.py 5 1 -p para_qtp.ini
#${eo_dir}/thawslumpScripts/test_kfold_figure.sh 5 2

# test svm classification
#${eo_dir}/thawslumpScripts/test_svm_classify.sh
