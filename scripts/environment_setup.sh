#!/bin/bash

WORKING_DIR=$HOME
MAP_PROJ_DIR=$HOME/MERT
echo 'running'

# conda setup

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ${WORKING_DIR}/miniconda
source ${WORKING_DIR}/miniconda/bin/activate
conda init

wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sh cuda_11.3.1_465.19.01_linux.run --silent --toolkit --toolkitpath=${WORKING_DIR}/cuda-11.3 --no-opengl-libs  #  --driver 
# should be placed at
# /usr/local/cuda
# /usr/local/cuda-11.3
export PATH=${WORKING_DIR}/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${WORKING_DIR}/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=${WORKING_DIR}/cuda-11.3

echo export PATH=${WORKING_DIR}/cuda-11.3/bin${PATH:+:${PATH}} >> ~/.bashrc
echo export LD_LIBRARY_PATH=${WORKING_DIR}/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} >> ~/.bashrc
echo export CUDA_HOME=${WORKING_DIR}/cuda-11.3 >> ~/.bashrc
nvcc -V

# pytorch
conda create -y -n map python=3.8
conda activate map
# pytorch                   1.12.1          py3.8_cuda11.3_cudnn8.3.2_0 
# pytorch-mutex             1.0                        cuda    pytorch
# torchaudio                0.12.1               py38_cu113    pytorch
# conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip3 install torch torchvision torchaudio
python -c 'import torch; print(torch.cuda.is_available())'


# NCCL setup

cd ${WORKING_DIR}
git clone https://github.com/NVIDIA/nccl.git
cd nccl

make -j12  src.build

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${WORKING_DIR}/nccl/build/lib
export PATH=$PATH:${WORKING_DIR}/nccl/build/bin
export NCCL_HOME=${WORKING_DIR}/nccl/build

echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${WORKING_DIR}/nccl/build/lib >> ~/.bashrc
echo export PATH=$PATH:${WORKING_DIR}/nccl/build/bin >> ~/.bashrc
echo export NCCL_HOME=${WORKING_DIR}/nccl/build  >> ~/.bashrc

python -c "import torch;print(torch.cuda.nccl.version())"

cd ${WORKING_DIR}
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j12
# 
ngpus=4
./build/all_reduce_perf -b 8 -e 256M -f 2 -g ${ngpus} 


# conda install cython
pip install --upgrade pip

mkdir ${MAP_PROJ_DIR}/src/fairseq
cd ${MAP_PROJ_DIR}/src
# fairseq                   0.12.2                   pypi_0    pypi
git clone https://github.com/pytorch/fairseq
# cd fairseq
cd ${MAP_PROJ_DIR}/src/fairseq
pip install --editable ./
# to solve pip subprocess error
# pip install --no-build-isolation ./ 


# apex  for half-precision training
pip3 install packaging

cd ${WORKING_DIR}
git clone https://github.com/NVIDIA/apex
cd apex
# deprecated modification
# sed -i '32 a \ \ \ \ \ \ \ \ return 0' setup.py # ignore cuda version
pip install -v --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./


conda install -y -c conda-forge pysoundfile libsndfile librosa ffmpeg
pip install pyarrow pydub npy-append-array tensorboardX scikit-learn pandas nnAudio wandb
# pip install --upgrade numpy==1.23


wandb login


# for fsdp 
pip install fairscale
pip install deepspeed
ds_report


# for Encodec data preparation
pip install -U encodec