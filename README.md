# MSVF: Multi-task Structure Variation Filter with Transfer Learning in High-throughput Sequencing
The single molecule real-time sequencing technologies, such as PacBio and Nanopore, have higher throughput and produce longer reads, which promote the discovery of more structure variations that cannot be discovered by the second-generation sequencing data.
However, compared with the second-generation sequencing data, the PacBio data lacks paired-end sequencing information, making traditional structure variations filter fail to process the new data. To solve this problem, this paper proposes a universal multi-tasking structure variation filtering model MSVF.
MSVF adopts the CIGAR string defined in SAM format. CIGAR is not limited by sequencing technology or alignment algorithms, so MSVF is suitable for not only the second-generation but also the third-generation sequencing data. Moreover, CIGAR string preserves the complete sequence alignment information, which makes MSVF a highly precise model.
Besides, MSVF uses deep learning methods, making it supports more structure variation types, including deletion and insertion.
We trained and tested the models on the open-access NCBI datasets. The experiments proved that ShuffleNet, MobileNet, ResNet transfer learning models achieve better classification results on SVs task. 
# Data description

| File name  | Description |
| ------------- | ------------- |
| Uniprot_ARG.fasta  | Antibiotic resistance genes in the UNIPROT database were used for model training and validation  |
| Uniprot_ARG_ind.fasta  | Independent antibiotic resistance genes in the UNIPROT database  |
| Uniprot_VF.fasta  | Virulence factors in the UNIPROT database were used for model training and validation  |
| Uniprot_VF_ind.fasta| Independent virulence factors in the UNIPROT database |
| Uniprot_NS.fasta| Negative genes (neither VFs nor ARGs) in the UNIPROT database were used for model training and validation| 
| Uniprot_NS_ind.fasta|  Independent negative genes (neither VFs nor ARGs) in the UNIPROT database| 
| Uniprot_ARG+VF+NS.fasta|  Total of 3 types of genes in the UNIPROT database were used for model training and validation| 
| Label_ARG+VF+NS.csv|  Label of 3 types of genes in the UNIPROT database were used for model training and validation| 
| Database_GENE.zip| The remaining database genes were used as known ARGs and VFs| 

# Requirements
HyperVR is tested to work under:

Python 3.8

Tensorflow 2.8.0

Keras 2.8.0

numpy 1.21.2

sklearn 1.1.1

Xgboost 1.5.2

# Quick start
To reproduce our results:

1, Download uniref dataset and install the required toolkit
```
cd tools/ncbi-blast && wget -c https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.13.0+-x64-linux.tar.gz && tar -zxvf ncbi-blast-2.13.0+-x64-linux.tar.gz

cd tools/diamond && wget -c https://github.com/bbuchfink/diamond/releases/download/v2.0.5/diamond-linux64.tar.gz && tar -zxvf diamond-linux64.tar.gz 

cd tools/uniref50 && wget -c https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz && tar -zxvf uniref50.fasta.gz

/tools/ncbi-blast/ncbi-blast-2.13.0+/bin/makeblastdb -dbtype prot -in uniref50.fasta -input_type fasta -parse_seqids -out uniref50_blast
```
2, Run generate_pssm_profile.py to generate pssm profiles for each gene sequence, the options are:
```
python src/generate_pssm_profile.py --file /data/Uniprot_ARG+VF+NS.fasta --blastpgp /tools/ncbi-blast/ncbi-blast-2.13.0+/bin --db /tools/uniref50/uniref50_blast --outdir /src/pssm_profile

--file: protein sequence file in fasta format
--blastpgp: the path of NCBI psiblast program
--db: the path of unief50 database
--outdir: the path of out dir
```
3, Run generate_bitscore.py to generate bitscore features for each gene sequence, the options are:
```
python src/generate_bitscore.py --file /data/Uniprot_ARG+VF+NS.fasta --db_file /data/Database_GENE.fasta --diamond_path /tools/diamond/diamond --outdir /src/bitscore

--file: protein sequence file in fasta format
--db_file: protein sequence file in fasta format
--diamond_path: the path of diamond program
--outdir: the path of out dir
```
4, Run generate_features/main.py to generate statistical gene sequence patterns for each gene sequence, the options are:
```
python /src/generate_features/main.py --file /data/Uniprot_ARG+VF+NS.fasta --type AAC --out /src/AAC_encoding.tsv
python /src/generate_features/main.py --file /data/Uniprot_ARG+VF+NS.fasta --type DPC --out /src/DPC_encoding.tsv
python /src/generate_features/main.py --file /data/Uniprot_ARG+VF+NS.fasta --type DDE --out /src/DDE_encoding.tsv
python /src/generate_features/main.py --file /data/Uniprot_ARG+VF+NS.fasta --type PAAC --out /src/PAAC_encoding.tsv
python /src/generate_features/main.py --file /data/Uniprot_ARG+VF+NS.fasta --type QSOrder --out /src/QSOrder_encoding.tsv
python /src/generate_features/main.py --file /data/Uniprot_ARG+VF+NS.fasta --type OHE --out /src/OHE_encoding.tsv
python /src/generate_features/main.py --file /data/Uniprot_ARG+VF+NS.fasta --type PSSMC --path /src/pssm_profiles --out /src/PSSMC_encoding.tsv
python /src/generate_features/main.py --file /data/Uniprot_ARG+VF+NS.fasta --type RPMPSSM --path /src/pssm_profiles --out /src/RPMPSSM_encoding.tsv
python /src/generate_features/main.py --file /data/Uniprot_ARG+VF+NS.fasta --type AADPPSSM --path /src/pssm_profiles --out /src/AADPPSSM_encoding.tsv

--file: input protein sequence file in fasta format
--type: the encoding type
--path: data file path used for 'PSSMC', 'RPMPSSM' and 'AADPPSSM' encodings
--out: the generated descriptor file
```
5, Run HyperVR_cv/main.py to train and validate model by 5-fold cv, the options are:
```
# please do not change the feature and label file name
python /src/HyperVR_cv/main.py --feature_path /src --label_path /data

--feature_path: the feature file path
--label_path: the label file path

```
# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

# Contacts
If you have any questions or comments, please feel free to email: byj@hnu.edu.cn.
