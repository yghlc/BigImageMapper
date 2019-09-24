#!/usr/bin/env bash

# copy selected file stored in a txt file.

txt=inf_image_list.txt
for file in $(cat $txt); do

    scp $file $hlcR3_host:${PWD}/.
    scp ${file}.ovr $hlcR3_host:${PWD}/.
done




