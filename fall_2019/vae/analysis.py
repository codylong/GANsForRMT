## things added to format for paper
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import pyplot as pltf
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
fsize = 20
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', size=fsize)  # controls default text sizes
plt.rc('axes', titlesize=20)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
plt.rc('legend', fontsize=14)  # legend fontsize
plt.rc('figure', titlesize=fsize)  # fontsize of the figure title

from torch import nn

class VAE(nn.Module):
    def __init__(self,h11):
        super(VAE, self).__init__()
        self.h11 = h11
        self.fc1 = nn.Linear(h11*h11, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, h11*h11)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.h11*self.h11))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


import os
import numpy as np
import seaborn as sns
# sns.set_palette("husl")
import matplotlib
import pandas as pd
import torch
import torchvision
from torchvision.utils import make_grid
from scipy.stats import wasserstein_distance
import torchvision.transforms as transforms
from torch.nn import functional as F


# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:800% !important; }</style>"))

def batch_upper_left(tensor, m):  # in: (batch_size,1,n,n) out: upper left (batch_size, 1, m, m)
    return torch.Tensor.narrow(torch.Tensor.narrow(tensor, 3, 0, m), 2, 0, m)


def histogram(df_column, title, xlabel='', kde=False, norm_hist=True, show=True, log10=True, ylim=None, xlim=None,
              filename='', dpi=500):
    sns.set_style("darkgrid")
    if log10:
        df_column = np.log10(df_column)
        df_column = df_column.replace([np.inf, -np.inf], np.nan)
        df_column = df_column.dropna()
        df_column = df_column.tolist()
        # print type(df_column)
        plot = sns.distplot(df_column, kde=kde, norm_hist=norm_hist, label=None)
    else:
        df_column = df_column.tolist()
        plot = sns.distplot(df_column, kde=kde, norm_hist=norm_hist, label=None)
    if title != '': plt.title(title)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    if xlabel != '': plt.xlabel(xlabel)  # xlabel(r'$\mathrm{Number\,\, of\,\, steps}$')
    if ylim != None: plt.ylim(ylim)
    if xlim != None: plt.xlim(xlim)
    if filename != '': plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()
    return plot


def histograms(df_columns, labels, kde=False, norm_hist=True, show_together=True, show_separate=False, log10=True):
    from matplotlib import rcParams
    rcParams['figure.figsize'] = 11.7, 8.27
    plt.legend(prop={'size': 12})
    for idx, df_column in enumerate(df_columns):
        histogram(df_column, labels[idx], kde=kde, norm_hist=norm_hist, show=show_separate, log10=log10)
    if show_together:
        plt.show()


def gen_matrices(GAN, noise=None, batchSize=64, nz=100):
    print "GANTYPE", type(GAN)
    if type(noise) == type(None):
        noise = torch.FloatTensor(batchSize, nz)
        #noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
    tensor = GAN.decode(noise)
    h11 = int(np.sqrt(tensor.shape[1]))
    tensor = tensor.view(tensor.shape[0], h11, h11)
    return tensor


def gen_wishart_matrices(h11, batchSize=64):
    A = torch.randn(batchSize, h11, h11)
    AT = torch.transpose(A, 1, 2)
    return torch.bmm(AT, A) / torch.tensor(1.0 * h11)


def ensemble_eigenvalues(data_tensor):  # data_tensor is batch of matrices
    eigs = []
    for matrix in data_tensor:
        img_size = matrix.shape[1]
        matrix = matrix.view(img_size, img_size).detach().numpy()
        eigs.extend(np.linalg.eig(matrix)[0])
    return eigs


def show_images(tensor, nrow=8, padding=2, dpi=500,
                normalize=False, range=None, scale_each=False, pad_value=0, scale_factor=1, title='', xlabel='',
                filename=''):
    sns.set_style("white")
    #tensor.data = tensor.data.mul(0.5).add(0.5)
    h11 = int(np.sqrt(tensor.shape[1]))
    tensor = tensor.view(tensor.shape[0],h11,h11)
    size = tensor.shape[2]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=scale_factor * size),
        transforms.ToTensor()
    ])
    tensor = [transform(x_) for x_ in tensor]
    img = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                                      normalize=normalize, range=range, scale_each=scale_each)
    npimg = img.detach().numpy()
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.yticks([])
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    if xlabel != '': plt.xlabel(xlabel)  # xlabel(r'$\mathrm{Number\,\, of\,\, steps}$')
    if title != '': plt.title(title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    if filename != '': plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def show_GAN_image_sequence(GANlist, batchSize=64, nz=100, scale_factor=1, dpi=500):
    fixed_noise = torch.FloatTensor(batchSize, nz)
    #fixed_noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
    for h11, epoch, GAN in GANlist:
        print "(h11, epoch):", (h11, epoch, nz)
        gen_matrices(GAN, noise=fixed_noise, nz=nz)


        show_images(gen_matrices(GAN, noise = fixed_noise, nz=nz), scale_factor = scale_factor, \
                    dpi = dpi, xlabel = r'$\mathrm{Epoch \,\,} ' + str(epoch) +'$', \
                   filename = 'images_for_paper/h11_' + str(h11) + '_nz_' +str(nz)+'_epoch_'+str(epoch)+'.png',
                   title =  r'$h^{11} = ' + str(h11) + ',\,\,\, n_z = ' + str(nz) +"$")

def show_GAN_histogram(GAN, h11, nz=100, dpi=500, noise=None, inverse_wishart=False, batchSize=10000, kde=False,
                       display_wishart=True, norm_hist=True, show=True, log10=True, ylim=None, xlim=None):
    data_tensor = gen_matrices(GAN, noise=noise, batchSize=batchSize, nz=nz)

    #data_tensor = batch_upper_left(data_tensor, h11)
    eigs = ensemble_eigenvalues(data_tensor)
    wishart_eigs = ensemble_eigenvalues(gen_wishart_matrices(h11, batchSize))
    if inverse_wishart:
        wishart_eigs = [1 / k for k in wishart_eigs]
    #
    # log_eigs, log_wishart_eigs, log_real_eigs = [k for k in np.log10(eigs) if k not in [np.nan, np.inf, -np.inf]], [k
    #                                                                                                                 for
    #                                                                                                                 k in
    #                                                                                                                 np.log10(
    #                                                                                                                     wishart_eigs)
    #                                                                                                                 if
    #                                                                                                                 k not in [
    #                                                                                                                     np.nan,
    #                                                                                                                     np.inf,
    #                                                                                                                     -np.inf]], [
    #                                                 k for k in np.log10(np.array(real_eig_dfs[h11])) if
    #                                                 k not in [np.nan, np.inf, -np.inf]]
    # nologdist, logdist = wasserstein_distance(eigs, real_eig_dfs[h11]), wasserstein_distance(log_eigs, log_real_eigs)
    # print "Wasserstein Distance GAN (no log, log):", nologdist, logdist
    # print "Wasserstein Distance Wishart (no log, log):", wasserstein_distance(wishart_eigs,
    #                                                                           real_eig_dfs[h11]), wasserstein_distance(
    #     log_wishart_eigs, log_real_eigs)
    # if log10:
    #     dist = logdist
    # else:
    #     dist = nologdist
    #
    # title = r'$h^{11} = ' + str(h11) + ',\,\,\, n_z = ' + str(nz) + ",\,\,\, \mathrm{\,\,\, \\\\ Distance = }" + str(
    #     round(dist, 2)) + "$"
    # xlabel = r'$\mathrm{log}_{10}(\mathrm{Eigenvalue})$'
    #
    # histogram(real_eig_dfs[h11], title=title, show=False, log10=log10, norm_hist=norm_hist)
    # if display_wishart:
    #     histogram(pd.DataFrame({h11: wishart_eigs})[h11], title=title, kde=kde, norm_hist=norm_hist, show=False,
    #               log10=log10)
    # histogram(pd.DataFrame({h11: eigs})[h11], title=title, kde=kde, norm_hist=norm_hist, show=show, log10=log10,
    #           ylim=ylim, xlim=xlim, \
    #           filename='images_for_paper/h11_' + str(h11) + '_nz_' + str(nz) + '_epoch_' + str(epoch) + '_plot.png',
    #           xlabel=xlabel, dpi=dpi)
    #
    # return wasserstein_distance(eigs, real_eig_dfs[h11]), wasserstein_distance(wishart_eigs,
    #                                                                            real_eig_dfs[h11]), wasserstein_distance(
    #     log_eigs, log_real_eigs)


# isize: imageSize, nz: size of latent z vector,ng
def load_DCGAN(filepath, h11=None, nz=100, nc=1, ngf=64, ngpu=0, n_extra_layers=0):
    print filepath
    # if h11 == None: h11 = int(filepath.split('/')[0].split('_')[2])
    epoch = int(filepath.split('/')[-1].split('_')[-1].split('.')[0])
    # isize = (h11-h11%16)+16
    vae = VAE(h11)
    VAE.load_state_dict(vae, torch.load(filepath, map_location='cpu'))
    return (h11, epoch, vae)

exp_path = "results_ks/"
h11 = 10
nz=20
gen_epochs = range(0,1001,20) # note after setting, only goes by 200
gen_epochs = [0,1000,2000]
print 'generator epochs to study:', gen_epochs
netG_paths = [exp_path + d for d in os.listdir(exp_path) if 'pth' in d]
print 'generator files:', netG_paths


h11_10_netGs = [load_DCGAN(p,h11=h11) for p in netG_paths]
h11_10_netGs = sorted(h11_10_netGs, key = lambda x: x[1])
#
# show_GAN_image_sequence(h11_10_netGs, nz=nz, scale_factor = 1, dpi=500)


plt.clf()
for _, epoch, netG in h11_10_netGs:
    print 'epoch:', epoch
    show_GAN_histogram(netG, 10,
                       batchSize=1000, nz=nz, log10=True, dpi=300, display_wishart=True, ylim=(0, 1), xlim=(-6, 2))

