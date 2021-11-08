# Compared to fix_h11 code, drawing some condition GAN inspiration from
# 
# https://github.com/malzantot/Pytorch-conditional-GANs

import numpy as np
import torch
import os
import pickle
import seaborn as sns
import torchvision
import platform
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import pyplot as pltf
import matplotlib as mpl
import copy
import pandas as pd
import argparse
from generator import *
from discriminator import *
from WGAN import *
from GAN import *
#from EigNN import *


def to_onehot(batch,num_digits):
    batch_size = batch.shape[0]
    onehot = torch.FloatTensor(batch_size,num_digits)
    onehot.zero_()
    onehot.scatter_(1,batch,1)
    return onehot

def load_metric_data(args):

    datasets, real_evals = {}, {}
    for h11 in args.h11s_test:
        isize = args.isize
        diff = isize-h11
        print('Loading data for h11 =', h11)
        ###
        # grab Kahler metrics for fixed number of geometries
        files_list = [d for d in os.listdir(args.dataroot) if "_" + str(h11) + "_" in d and 'K' in d]
        files_list = files_list[:args.num_geometries]

        raw_pickles = []
        for d in files_list:
            try:
                if args.py_version == 3:
                    raw_pickles.append(pickle.load(open(args.dataroot + d, 'rb'), encoding='latin1'))
                else:
                    raw_pickles.append(pickle.load(open(args.dataroot + d, 'r')))
            except ValueError:
                print("Error reading pickle:", d)

        padded_raw_pickles = [torch.nn.functional.pad(torch.FloatTensor(k), (0, diff, 0, diff), 'constant', 0).numpy() for k in raw_pickles]

        # create PyTorch dataset
        raw_data = torch.tensor(padded_raw_pickles).view(len(padded_raw_pickles), 1, isize, isize)
        datasets[h11] = padded_raw_pickles

        ###
        # now grab eigenvalues for geometries not trained on
        test_list = [d for d in os.listdir(args.dataroot) if"_" + str(h11) + "_" in d and 'K' in d and d not in files_list]
        test_list = [d[0:].replace('K', 'evals') for d in test_list]
        cur_evals = []
        for test_file in test_list:
            if args.py_version == 3:
                cur_evals.extend(pickle.load(open(args.dataroot + test_file, 'rb'), encoding='latin1'))
            else:
                cur_evals.extend(pickle.load(open(args.dataroot + test_file, 'r')))
        real_evals[h11] = cur_evals

    # create PyTorch dataloader that will be used for training
    data = []
    labels = []
    for h11 in args.h11s_train:
        print("Adding data to dataset for h11 = ", h11)
        data.extend(datasets[h11])
        labels.extend([[1.0 if k == h11-1 else 0.0 for k in range(args.isize)] for p in datasets[h11]])

    labels = torch.tensor(labels)
    dataset = torch.tensor(data).view(len(data),1,isize,isize)
    dataset = torch.utils.data.TensorDataset(dataset, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    return data_loader, datasets, real_evals, len(test_list)

def save_experiment_initial(args): # copies source file
    from shutil import copyfile
    import sys, os

    if args.exp == None:
        date = str(datetime.datetime.now())
        date_arr = date.split(" ")[0].split("-")
        time_str = date.split(" ")[1].replace(":","").split(".")[0]
        file_prefix = sys.argv[0].split('/')[-1].split(".")[0]
        outdir = "./experiments/exp_" + args.model_type + "_" + str(args.h11) + "_" + str(args.nz) + "_" + date_arr[1] + date_arr[2] + date_arr[0] + "_" + time_str
    else:
        outdir = args.exp
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    f = open(outdir + "/args.txt", "w")
    f.write(str(vars(args)))
    f.close()

    source_filename = "./" + sys.argv[0].split('/')[-1]
    source_filename_target = os.path.join(outdir,source_filename)
    copyfile(source_filename,source_filename_target)
    log_filename = os.path.join(outdir,"exp.log")
    f = open(log_filename,"w")
    sys.stdout = f

    return outdir

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default = "/home/jim/Documents/RandomBergman/KS4/data/")
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--h11', type=int, default = 10)
    parser.add_argument('--newsize', type=int, default = 40) # in case h11xh11 is upper left
    parser.add_argument('--num-geometries', type=int, default = 2500)
    parser.add_argument('--batch-size', type=int, default = 64)
    parser.add_argument('--nz', type=int, default = 15)
    parser.add_argument('--lr-D', type=float, default =.000005)
    parser.add_argument('--lr-G', type=float, default =.000005)
    parser.add_argument('--epochs', type=int, default = 1000)
    parser.add_argument('--label-hid', type=int, default = 500)
    parser.add_argument('--log-interval', type=int, default = 1)
    parser.add_argument('--plot-interval', type=int, default = 20)
    parser.add_argument('--n-critic-steps', type=int, default = 5)
    parser.add_argument('--test-batch-size', type=int, default= 5000)
    parser.add_argument('--model-type', type=str, default = 'WGAN_DCGAN', choices=['WGAN_FFNN', 'WGAN_DCGAN', 'GAN_FFNN', 'GAN_DCGAN', 'EigTarget_FFNN', 'EigTarget_DCGAN'])
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--show_plots', action='store_true', default=False)

    args = parser.parse_args()

    args.lr = min(args.lr_D, args.lr_G)
    args.diff = 0 # args.newsize - args.h11
    args.py_version = int(platform.python_version()[:1])
    args.h11s_train = [10,30]
    args.h11s_test = [10,20,30]
    args.h11s_train = [10,20]
    args.h11s_test = [10,20,30]
    args.isize = max(args.h11s_train+args.h11s_test)
    args.max_h11 = args.isize
    print("Image Size:", args.isize,"x",args.isize)

    args.outdir = "."
    if args.save:
        args.outdir = save_experiment_initial(args)
        print(datetime.datetime.now())
        print(args)

    data_loader, data_sets, real_eigs, num_test_geometries = load_metric_data(args)
    print(args)
    print("Model type:", args.model_type)
    print("Comparing to Eigenvalue distribution collected from", num_test_geometries, "geometries")

    ##
    # Print distribution data for real eigenvalues
    print("\n\nComparing real distributions via wass and log wass:")
    keys = list(real_eigs.keys())
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            k1, k2 = keys[i], keys[j]
            r1, r2 = real_eigs[k1], real_eigs[k2]
            l1, l2 = [np.log10(k) for k in r1], [np.log10(k) for k in r2]
            l1 = [k for k in l1 if not np.isnan(k) and k not in [np.nan, np.inf, -np.inf]]
            l2 = [k for k in l2 if not np.isnan(k) and k not in [np.nan, np.inf, -np.inf]]
            print("\t(h11_1,h11_2) = (%d,%d): %.3f, %.3f" % (k1,k2,wasserstein_distance(r1,r2),wasserstein_distance(l1,l2)))


    train_WGAN(args, cDCGAN_G(args.max_h11, args.nz, args.label_hid), cDCGAN_D(args.max_h11, args.nz, args.label_hid), data_loader, real_eigs, num_test_geometries)