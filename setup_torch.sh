#!/bin/bash -x

if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    conda install pytorch cpuonly -c pytorch # this will install the latest version by default
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
    NVCC_VERSION=`nvcc --version | grep release | awk '{print $5}' | sed 's/.$//'`
    if [ "$?" == 0 ]; then # previous command sucess
        echo nvcc_version=$NVCC_VERSION
    else
        echo please manually check your cudatoolkit version and install pytorch
        exit 1
    fi
    conda install pytorch cudatoolkit=$NVCC_VERSION -c pytorch # this will install the latest version by default
fi