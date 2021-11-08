from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import cPickle as pickle
import numpy as np
import seaborn as sns 
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt

from VAE import VAE

        
exp_path = "results_ks/"
h11 = 10
nz=20
gen_epochs = [0,1000,2000]
vae_paths = [exp_path + d for d in os.listdir(exp_path) if 'pth' in d]
#print('generator files:', vae_paths)


h11_10_vaes = []# [load_DCGAN(p,h11=h11) for p in netG_paths]
for filepath in vae_paths:
    epoch = int(filepath.split('/')[-1].split('_')[-1].split('.')[0])
    vae = VAE(h11)
    VAE.load_state_dict(vae, torch.load(filepath, map_location='cpu'))
    h11_10_vaes.append((h11, epoch, vae))  

h11_10_vaes = sorted(h11_10_vaes, key = lambda x: x[1])

def check_symmetric(a, rtol=1e-08, atol=1e-12):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

for h11, epoch, vae in h11_10_vaes:
    print(epoch)
    sample = torch.randn(64,20)
    sample = vae.decode(sample).cpu()
    #save_image(sample.view(64, 1, h11, h11),'analysis_results_ks/sample_' + str(epoch) + '.png')
    
    eigs = []
    for matrix in sample:
        matrix = matrix.view(10,10).detach().numpy()
        matrix = (matrix + np.transpose(matrix))/2
        print(check_symmetric(matrix))
        print(matrix.shape)    
        eigs.extend(np.linalg.eig(matrix)[0])
    for eig in eigs:
        print(eig)
        assert eig >= 0
    sns.distplot(pd.DataFrame({h11: np.log10(eigs)})[h11]) 
    plt.show()
# #
# # show_GAN_image_sequence(h11_10_netGs, nz=nz, scale_factor = 1, dpi=500)


# plt.clf()
# for _, epoch, netG in h11_10_netGs:
#     print 'epoch:', epoch
#     show_GAN_histogram(netG, 10,
#                        batchSize=1000, nz=nz, log10=True, dpi=300, display_wishart=True, ylim=(0, 1), xlim=(-6, 2))

