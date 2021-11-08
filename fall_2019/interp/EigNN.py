from generator import *
from analysis import *
from torch.autograd import Variable
import copy

from torchsearchsorted import searchsorted
import numpy as np
from scipy.stats import wasserstein_distance
import seaborn as sns
from matplotlib import pyplot as plt
import torch

# set up test nn
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def remove_nan(t):
    nan = torch.isnan(t[0])
    o = torch.tensor([])
    for idx, p in enumerate(nan):
        # print p, o
        if p == 0:
            o = torch.cat((o, torch.tensor([t[0][idx]])))
        else:
            o = torch.cat((o, torch.tensor([0.0])))
    return torch.tensor([list(o)])


def _cdf_distance(p, u_values, v_values):
    # print "input", u_values, v_values
    # print '\nsorter'
    u_sorter = torch.argsort(u_values).to(device)
    v_sorter = torch.argsort(v_values).to(device)

    # print 'allvals'
    all_values = torch.cat((u_values, v_values))
    all_values, _ = all_values.sort()
    # print'av',all_values

    #     # Compute the differences between pairs of successive values of u and v.
    deltas = all_values[1:] - all_values[:-1]  # replaces call to np.diff
    #     print 'searchsorted'
    #     u_cdf_indices = searchsorted(torch.tensor([list([u_values[u_sorter]][0])]), torch.tensor([list([all_values[:-1]][0])])).to(device)
    #     v_cdf_indices = searchsorted(torch.tensor([list([v_values[v_sorter]][0])]), torch.tensor([list([all_values[:-1]][0])])).to(device)
    #     #print 'ucdfind', u_cdf_indices

    u1, u2 = remove_nan(torch.tensor([list([u_values[u_sorter]][0])])), remove_nan(
        torch.tensor([list([all_values[:-1]][0])]))
    u_cdf_indices = searchsorted(u1, u2).to(device)
    v1, v2 = remove_nan(torch.tensor([list([v_values[v_sorter]][0])])), remove_nan(
        torch.tensor([list([all_values[:-1]][0])]))
    v_cdf_indices = searchsorted(v1, v2).to(device)

    # print 'cdf'
    # Calculate the CDFs of u and v using their weights, if specified.
    u_cdf = u_cdf_indices / u_values.shape[0]
    v_cdf = v_cdf_indices / v_values.shape[0]

    # print 'uv', u_cdf, v_cdf

    # print 'check1', deltas
    # print 'sum and multiply'

    return torch.sum(torch.abs(u_cdf - v_cdf) * deltas)
    return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    # print 'return from loss'
    return torch.pow(torch.sum(torch.pow(torch.abs(u_cdf - v_cdf), p) * deltas), 1 / p)


def custom_wasserstein_distance(u, v):
    return _cdf_distance(1, u, v)


def to_var(x):
    # first move to GPU, if necessary
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def train_EigNN(args, generator, data_loader, real_eigs, num_test_geometries, log_eigs = True):
    if torch.cuda.is_available():
        device = 'cuda'
        generator.cuda()

    real_eigs_tensor = torch.tensor(real_eigs)

    real_log_eigs_tensor = [np.log10(k) for k in real_eigs]
    real_log_eigs_tensor = torch.tensor([k for k in real_log_eigs_tensor if not np.isnan(k) and k not in [np.nan, np.inf, -np.inf]]).to(device)

    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.lr_G)

    EigNN_list = []


    try:
        for epoch in range(args.epochs + 1):
            generator.zero_grad()

            # draw random vars, generate fakes, and compute fake errors
            z = to_var(torch.randn(args.num_geometries, args.nz))
            fake_images = generator(z)
            eigs = torch.tensor([]).to(device)
            for fake in fake_images:
                eigs = torch.cat((eigs,torch.symeig(fake.view(args.h11,args.h11),eigenvectors=True)[0]))
            #eigs = eigs.view(eigs.shape[0]*eigs.shape[1])
            print('begincomp')
            if log_eigs:
                eigs = torch.log10(eigs)
                generator_loss = custom_wasserstein_distance(eigs,real_log_eigs_tensor)
            else:
                generator_loss = custom_wasserstein_distance(eigs,real_eigs_tensor)
            print('endcomp')

            generator_loss.backward()
            generator_optimizer.step()

            ###
            # Print progress as training occurs
            ###

            if (epoch) % args.log_interval == 0:
                EigNN_list.append((args.h11, epoch, copy.deepcopy(generator).cpu()))
                h11, _, netG = EigNN_list[-1]
                wass, wass_log = test_generator(netG, args, real_eigs)

                print('Epoch [%d/%d], '
                      'g_loss: %.4f, Wasserstein(eigs,real_eigs): %.2f, Wasserstein(log_eigs,log_real_eigs): %.2f'
                      % (epoch,
                         args.epochs,
                         generator_loss.data,
                         wass,
                         wass_log)
                      )

                show_GAN_histogram(netG, epoch, args.h11, args, real_eigs,
                                 batchSize=10000, nz=args.nz, log10=True, dpi=300, display_wishart=True, ylim=(0, 1),
                                 xlim=(-6, 2))

    except KeyboardInterrupt:
        print("Training ended via keyboard interrupt.")