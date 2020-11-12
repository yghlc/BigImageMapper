FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 
MAINTAINER Lingcao Huang

# docker for running tensorflow on Centos (ITSC services)
# This docker only provides a OS (ubuntu 16.04) for running our scirpt
# It will mount ~/codes, ~/programs, ~/Data
# The python, tensorflow, and other depends are in ~/programs, we will copy them from Cry03
# Python is ~/programs/anaconda2/bin/


RUN apt-get update && apt-get install -y --no-install-recommends \
        git
#        build-essential \
#        pkg-config \
#        m4 \
#        g++ \
#        zlib1g-dev \
#        python \
#        python3-dev \
#        vim

#        zip \
#        unzip \
#        git \
#	    wget \
#	    gedit

# isce dependency
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        libgmp-dev libmpfr-dev libmpc-dev libc6-dev-i386 \
#        gfortran \
#        fftw-dev \
#        libfftw3-dev \
#        sqlite3 libsqlite3-dev \
#        libmotif-dev \
#        libxt-dev  \
#        sudo \
#        scons \
#        libhdf5-dev \
#        cython3 \
#        libatlas-dev libatlas-base-dev


# Clean-up
#RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# need to create a user: hlc
#RUN useradd -m hlc && echo "hlc:hlc123" | chpasswd && adduser hlc sudo
#USER hlc
CMD /bin/bash
#WORKDIR /home/hlc

#set env


















