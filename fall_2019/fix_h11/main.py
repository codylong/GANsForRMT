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



def load_metric_data(args):

    ###
    # grab Kahler metrics for fixed number of geometries
    files_list = [d for d in os.listdir(args.dataroot) if "_" + str(args.h11) + "_" in d and 'K' in d]
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

    padded_raw_pickles = [torch.nn.functional.pad(torch.FloatTensor(k), (0, args.diff, 0, args.diff), 'constant', 0).numpy() for k in raw_pickles]

    # create PyTorch dataset
    raw_data = torch.tensor(padded_raw_pickles).view(args.num_geometries, 1, args.h11, args.h11)
    dataset = torch.utils.data.TensorDataset(raw_data)

    # create PyTorch dataloader that will be used for training
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    ###
    # now grab eigenvalues for geometries not trained on
    test_list = [d for d in os.listdir(args.dataroot) if"_" + str(args.h11) + "_" in d and 'K' in d and d not in files_list]
    test_list = [d[0:].replace('K', 'evals') for d in test_list]
    cur_evals = []
    for test_file in test_list:
        if args.py_version == 3:
            cur_evals.extend(pickle.load(open(args.dataroot + test_file, 'rb'), encoding='latin1'))
        else:
            cur_evals.extend(pickle.load(open(args.dataroot + test_file, 'r')))

    return data_loader, cur_evals, len(test_list)

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
    parser.add_argument('--log-interval', type=int, default = 1)
    parser.add_argument('--plot-interval', type=int, default = 20)
    parser.add_argument('--n-critic-steps', type=int, default = 5)
    parser.add_argument('--test-batch-size', type=int, default= 10000)
    parser.add_argument('--model-type', type=str, default = 'WGAN_DCGAN', choices=['WGAN_FFNN', 'WGAN_DCGAN', 'GAN_FFNN', 'GAN_DCGAN', 'EigTarget_FFNN', 'EigTarget_DCGAN'])
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--show_plots', action='store_true', default=False)

    args = parser.parse_args()

    args.lr = min(args.lr_D, args.lr_G)
    args.diff = 0 # args.newsize - args.h11
    args.py_version = int(platform.python_version()[:1])

    args.outdir = "."
    if args.save:
        args.outdir = save_experiment_initial(args)
        print(datetime.datetime.now())
        print(args)

    data_loader, real_eigs, num_test_geometries = load_metric_data(args)
    print(args)
    print("Model type:", args.model_type)
    print("Coomparing to Eigenvalue distribution collected from", num_test_geometries, "geometries")

    if args.model_type == "WGAN_FFNN":
        critic = nn.Sequential(
            nn.Linear(args.h11 * args.h11, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        train_WGAN(args, symmetric_G_FFNN(args), critic, data_loader, real_eigs, num_test_geometries)
    elif args.model_type == "WGAN_DCGAN":
        train_WGAN(args, DCGAN_G(args.h11, args.nz), DCGAN_D(args.h11, args.nz), data_loader, real_eigs, num_test_geometries)
    elif args.model_type == "GAN_FFNN":
        D = nn.Sequential(
            nn.Linear(args.h11 * args.h11, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        train_GAN(args, symmetric_G_FFNN(args), D, data_loader, real_eigs, num_test_geometries)
    elif args.model_type == "GAN_DCGAN":
        train_GAN(args, DCGAN_G(args.h11, args.nz), DCGAN_D_Sigmoid(args.h11, args.nz), data_loader, real_eigs, num_test_geometries)
    elif args.model_type == 'EigTarget_FFNN':
        train_EigNN(args, symmetric_G_FFNN(args), data_loader, real_eigs, num_test_geometries)
