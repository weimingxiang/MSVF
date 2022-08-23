# MSVF: Multi-task Structure Variation Filter with Transfer Learning in High-throughput Sequencing

## Introduction
The single molecule real-time sequencing technologies, such as PacBio and Nanopore, have higher throughput and produce longer reads, which promote the discovery of more structure variations that cannot be discovered by the second-generation sequencing data.
However, compared with the second-generation sequencing data, the PacBio data lacks paired-end sequencing information, making traditional structure variations filter fail to process the new data. To solve this problem, this paper proposes a universal multi-tasking structure variation filtering model MSVF.
MSVF adopts the CIGAR string defined in SAM format. CIGAR is not limited by sequencing technology or alignment algorithms, so MSVF is suitable for not only the second-generation but also the third-generation sequencing data. Moreover, CIGAR string preserves the complete sequence alignment information, which makes MSVF a highly precise model.
Besides, MSVF uses deep learning methods, making it supports more structure variation types, including deletion and insertion.
We trained and tested the models on the open-access NCBI datasets. The experiments proved that ShuffleNet, MobileNet, ResNet transfer learning models achieve better classification results on SVs task.

# Requirements
MSVF is tested to work under:

* Python 3.6

* pytorch 1.10.2

* pysam 0.15.4

* numpy 1.19.5

* scikit-learn 0.19.2

* torchvision 0.11.3

* tensorboard 2.8.0

# Quick start

## Environment by using anaconda and pip
```shell
conda create -n MSVF python=3.6

conda install pytorch torchvision torchaudio cudatoolkit -c pytorch -c conda-forge -y
conda install pytorch-lightning=1.5.10 -c conda-forge -y
pip install ray[tune] -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install redis -y
conda install scikit-learn -y
conda install matplotlib -y


conda install -c conda-forge tensorboardx tensorboard==1.15 -y
conda install scipy -y
```

## Code and Data
```shell
git clone git@github.com:weimingxiang/MSVF.git
```

Download all dataset from https://cowtransfer.com/s/cddf954624b247.


## Predict


# Usage
# License
This source code is licensed under the GPL license found in the LICENSE file in the root directory of this source tree.

# Contacts
If you have any questions or comments, please feel free to email: wmxiang@hnu.edu.cn.
