#!/usr/bin/env bash

# bash create new region defined parameter files (ini)

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 20 June, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

for dd in $(ls -d *_1); do

  echo $dd
  out=${dd}.zip
  if [ -f ${out} ]; then
    echo ${out} exist, skip
    continue
  fi
  # only add post*.shp and ini files
  zip -r ${out} ${dd}/*post* ${dd}/*.ini

done

