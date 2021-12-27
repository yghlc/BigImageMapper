#!/bin/bash

mapdir=/home/lihu9680/Data/Arctic/canada_arctic/autoMapping

uist_mapdir=/home/lhuang/Data/Arctic/canada_arctic/autoMapping

function remote_copy(){

    fo=$1   #folder
    machine=$2
    sshhost=$3
    mapdir=$4

    save_dir=${fo}_${machine}
    mkdir -p ${save_dir}
    cd ${save_dir}
        rdirs=$(ssh $sshhost "ls -d ${mapdir}/${fo}/GAN_*")
        rdirs_2=$(ssh $sshhost "ls -d ${mapdir}/${fo}/cycle_gan_*")  # if cycle_gan exist, it will replace GAN_*
        #echo $rdirs
        for rdir in ${rdirs} ${rdirs_2}; do
            echo $rdir
            ff=$(basename $rdir)
            echo $ff
            mkdir -p ${ff}
            cd ${ff}
                scp -p $sshhost:${rdir}/subImages_translate/*.tif .
            cd ..

        done
    
    cd ..
}

# exp 14, test GAN for time domain
remote_copy multiArea_deeplabV3+_10 tesia $tesia_host ${mapdir}
remote_copy multiArea_deeplabV3+_11 tesia $tesia_host ${mapdir}
remote_copy multiArea_deeplabV3+_12 uist $uist_host ${uist_mapdir}

# exp15, two source region, one Target region,
remote_copy multiArea_deeplabV3+_13 tesia $tesia_host ${mapdir}
remote_copy multiArea_deeplabV3+_14 tesia $tesia_host ${mapdir}
remote_copy multiArea_deeplabV3+_15 uist $uist_host ${uist_mapdir}

# exp16, split each region into two
remote_copy multiArea_deeplabV3+_16 tesia $tesia_host  ${mapdir}
remote_copy multiArea_deeplabV3+_17 tesia $tesia_host ${mapdir}
remote_copy multiArea_deeplabV3+_16 uist $uist_host ${uist_mapdir}

     

