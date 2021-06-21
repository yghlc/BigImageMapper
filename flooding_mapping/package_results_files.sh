#!/usr/bin/env bash

# bash create new region defined parameter files (ini)

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 20 June, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

for dd in $(ls -d *_1); do

  echo $dd
  # only add post*.shp and ini files
  zip -r ${dd}.zip ${dd}/*post* ${dd}/*.ini

done

