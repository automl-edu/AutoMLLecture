#!/bin/bash

# install dependencies
if [ ! -d "$dependencies" ]; then
	mkdir dependencies
fi	

cd dependencies

git clone https://github.com/google-research/nasbench
cd nasbench
pip install -e .
cd ..

git clone https://github.com/automl/nas_benchmarks.git
cd nas_benchmarks
python setup.py install
cd ../..

# download tabular benchmark
if [ ! -d "$benchmark" ]; then
	mkdir benchmark
fi

cd benchmark
wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
cd ..

pip install -r requirements.txt
