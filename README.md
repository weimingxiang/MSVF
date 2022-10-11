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

## Code and Data
```shell
git clone git@github.com:weimingxiang/MSVF.git
```

Download all dataset from https://cowtransfer.com/s/cddf954624b247. (You can also choose the sample data in the code)

## Model that have been trained
| File name  | URL |
| ------------- | ------------- |
| mnasnet1_0.ckpt  | https://github.com/weimingxiang/MSVF/releases/download/model/mnasnet1_0.ckpt  |
| mobilenet_v2.ckpt  | https://github.com/weimingxiang/MSVF/releases/download/model/mobilenet_v2.ckpt  |
| shufflenet_v2_x1_0.ckpt  | https://github.com/weimingxiang/MSVF/releases/download/model/shufflenet_v2_x1_0.ckpt  |
| resnet34.ckpt  | https://github.com/weimingxiang/MSVF/releases/download/model/resnet34.ckpt  |
| resnet50.ckpt  | https://github.com/weimingxiang/MSVF/releases/download/model/resnet50.ckpt  |
| resnet101.ckpt  | https://github.com/weimingxiang/MSVF/releases/download/model/resnet101.ckpt  |
| resnet152.ckpt  | https://github.com/weimingxiang/MSVF/releases/download/model/resnet152.ckpt  |

## Environment by using anaconda and pip
```shell
conda create -n MSVF python=3.6 -y
conda activate MSVF
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch -y
conda install pytorch-lightning=1.5.10 -c conda-forge -y
pip install ray[tune] -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install redis -y
conda install scikit-learn -y
conda install matplotlib -y
conda install samtools -y
conda install pudb -y
```

## Predict
```python
python predict.py selected_model
```
# Usage

## Easy to train
```
python simple_train.py selected_model
```
## Train
```
wget https://ftp.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NA12878_PacBio_MtSinai/sorted_final_merged.bam
parallel  samtools index ::: *.bam
wget https://ftp.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NA12878_PacBio_MtSinai/NA12878.sorted.vcf.gz
tar -xzvf NA12878.sorted.vcf.gz
python vcf_data_process.py
python bam2depth.py
python train.py
```

# License
This source code is licensed under the GPL license found in the LICENSE file in the root directory of this source tree.

# Contacts
If you have any questions or comments, please feel free to email: wmxiang@hnu.edu.cn.
