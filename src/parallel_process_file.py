import argparse
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
import utilities as ut
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.multiprocessing.set_sharing_strategy('file_system')

seed_everything(2022)

data_dir = "../data/"

bam_path = data_dir + "sorted_final_merged.bam"

ins_vcf_filename = data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = data_dir + "delete_result_data.csv.vcf"

hight = 224
resize = torchvision.transforms.Resize([hight, hight])


def p(sum_data):
    chromosome, chr_len = sum_data

    # copy begin

    print("deal " + chromosome)

    ins_position = torch.load(
        data_dir + 'position/' + chromosome + '/insert' + '.pt')
    del_position = torch.load(
        data_dir + 'position/' + chromosome + '/delete' + '.pt')
    n_position = torch.load(data_dir + 'position/' +
                            chromosome + '/negative' + '.pt')

    print("cigar start")
    if os.path.exits(ins_cigar_img, save_path + '/negative_cigar_new_img' + '.pt'):
        return

    ins_cigar_img = torch.empty(len(ins_position), 4, hight, hight)
    del_cigar_img = torch.empty(len(del_position), 4, hight, hight)
    negative_cigar_img = torch.empty(len(n_position), 4, hight, hight)

    for i, b_e in enumerate(ins_position):
        # f positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
        zoom = 1
        fail = 1
        while fail:
            try:
                fail = 0
                ins_cigar_img[i] = ut.cigar_new_img_single_optimal(
                    bam_path, chromosome, b_e[0], b_e[1], zoom)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
                print("Exception cigar_img_single_optimal " + str(zoom))
        #     try:
        #         positive_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
        #     except Exception as e:
        #         print(e)
        #         print("Exception cigar_img_single_optimal_time2sapce")
        #         try:
        #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
        #         except Exception as e:
        #             print(e)
        #             print("Exception cigar_img_single_optimal_time3sapce")
        #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])

        print("===== finish(ins_cigar_img) " + chromosome + " " + str(i))

    for i, b_e in enumerate(del_position):
        # f positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
        zoom = 1
        fail = 1
        while fail:
            try:
                fail = 0
                del_cigar_img[i] = ut.cigar_new_img_single_optimal(
                    bam_path, chromosome, b_e[0], b_e[1], zoom)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
                print("Exception cigar_img_single_optimal " + str(zoom))
        #     try:
        #         positive_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
        #     except Exception as e:
        #         print(e)
        #         print("Exception cigar_img_single_optimal_time2sapce")
        #         try:
        #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
        #         except Exception as e:
        #             print(e)
        #             print("Exception cigar_img_single_optimal_time3sapce")
        #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])

        print("===== finish(del_position) " + chromosome + " " + str(i))

    for i, b_e in enumerate(n_position):
        # f negative_cigar_img = torch.cat((negative_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
        zoom = 1

        fail = 1
        while fail:
            try:
                fail = 0
                negative_cigar_img[i] = ut.cigar_new_img_single_optimal(
                    bam_path, chromosome, b_e[0], b_e[1], zoom)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
                print("Exception cigar_img_single_optimal " + str(zoom))

            # try:
            #     negative_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
            # except Exception as e:
            #     print(e)
            #     print("Exception cigar_img_single_optimal_time2sapce")
            #     try:
            #         negative_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
            #     except Exception as e:
            #         print(e)
            #         print("Exception cigar_img_single_optimal_time3sapce")
            #         negative_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])

        print("===== finish(n_position) " + chromosome + " " + str(i))

    save_path = data_dir + 'image/' + chromosome

    torch.save(ins_cigar_img, save_path + '/ins_cigar_new_img' + '.pt')
    torch.save(del_cigar_img, save_path + '/del_cigar_new_img' + '.pt')
    torch.save(negative_cigar_img, save_path +
               '/negative_cigar_new_img' + '.pt')
    print("cigar end")


def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    help = "The path of address"
    parser.add_argument('--chr', help=help)
    parser.add_argument('--len', help=help)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # print(args.chr)
    # print(type(args.chr))
    p([args.chr, int(args.len)])
