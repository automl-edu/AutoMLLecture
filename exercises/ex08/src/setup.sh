#!/bin/bash

# needed for tensorboard compatibility
pip install --upgrade setuptools
pip install -r requirements.txt

# get the python version
v="$(python -V)"
echo $v
v="${v//[^0-9]/}"
v="${v:0:2}"

# install pytorch-cpu
pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp${v}-cp${v}m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp${v}-cp${v}m-linux_x86_64.whl

# install dependencies
if [ ! -d "$dependencies" ]; then
	mkdir dependencies
fi	

cd dependencies

git clone https://github.com/google-research/nasbench
cd nasbench
pip install -e .
cd ..

# download tabular benchmark
if [ ! -d "$benchmark" ]; then
	mkdir benchmark
fi

cd benchmark
wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
cd ..

