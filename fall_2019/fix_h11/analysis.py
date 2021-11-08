import torch
import numpy as np
from scipy.stats import wasserstein_distance
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import datetime

def gen_wishart_matrices(h11, batchSize = 64):
    A = torch.randn(batchSize,h11,h11)
    AT = torch.transpose(A,1,2)
    return (torch.bmm(AT,A)/torch.tensor(1.0*h11)).view(A.shape[0],h11*h11)

def ensemble_eigenvalues(data_tensor): # data_tensor is batch of matrices
    eigs, s = [], data_tensor.shape
    size = int(np.sqrt(s[1]))
    data_tensor = data_tensor.view(s[0],size,size)
    for matrix in data_tensor:
        img_size = matrix.shape[1]
        matrix = matrix.view(img_size,img_size).detach().numpy()
        eigs.extend(np.linalg.eig(matrix)[0])
    return eigs

def gen_matrices(generator, args, noise = None):
    if type(noise) == type(None):
        noise = torch.FloatTensor(args.test_batch_size, args.nz)
        noise.resize_(args.batch_size, args.nz).normal_(0, 1)
    
    return generator(noise) 

def test_generator(generator,args,real_eigs,plus_log_eigs=True):
    #print(datetime.datetime.now())
    fakes = gen_matrices(generator,args)
    #print(datetime.datetime.now())
    print(fakes.shape)
    eigs = ensemble_eigenvalues(fakes)
    eigs = [k for k in eigs if not np.isnan(k) and k not in [np.nan, np.inf, -np.inf]]

    if plus_log_eigs:
        log_eigs = [np.log10(k) for k in eigs]
        log_eigs = [k for k in log_eigs if not np.isnan(k) and k not in [np.nan, np.inf, -np.inf]]
        log_real_eigs = [np.log10(k) for k in real_eigs]
        log_real_eigs = [k for k in log_real_eigs if not np.isnan(k) and k not in [np.nan, np.inf, -np.inf]]

        return wasserstein_distance(eigs, real_eigs), wasserstein_distance(log_eigs, log_real_eigs)
    else:
        return wasserstein_distance(eigs,real_eigs), None

# from scipy.stats import wasserstein_distance

def histogram(df_column, title, xlabel = '', kde = False, norm_hist = True, show = True, log10 = True, ylim = None,xlim=None,filename = '',dpi=500):
    sns.set_style("darkgrid")
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    if log10:
        df_column = np.log10(df_column)
        df_column = df_column.replace([np.inf, -np.inf], np.nan)
        df_column = df_column.dropna()
        df_column = df_column.tolist()
        #print type(df_column)
        plot = sns.distplot(df_column, kde = kde, norm_hist = norm_hist, label = None)
    else:
        df_column = df_column.tolist()
        plot = sns.distplot(df_column, kde = kde, norm_hist = norm_hist, label = None)
    if title != '': plt.title(title)

    if xlabel != '': plt.xlabel(xlabel) #xlabel(r'$\mathrm{Number\,\, of\,\, steps}$')
    if ylim != None: plt.ylim(ylim)
    if xlim != None: plt.xlim(xlim)
    if filename != '': plt.savefig(filename,dpi=dpi,bbox_inches='tight',pad_inches=0.1)
    if show:
        plt.show()
    return plot

def histograms(df_columns, labels, kde = False, norm_hist = True, show_together = False, show_separate = False, log10 = True):
    from matplotlib import rcParams
    rcParams['figure.figsize'] = 11.7,8.27
    plt.legend(prop={'size':12})
    for idx, df_column in enumerate(df_columns):
        histogram(df_column, labels[idx], kde = kde, norm_hist = norm_hist, show = show_separate, log10 = log10)
    if show_together:
        plt.show()
        

def show_GAN_histogram(GAN, epoch, h11, args, real_eigs, nz = 100, dpi=500, noise = None, inverse_wishart = False, batchSize = 10000, kde = False, display_wishart = True, norm_hist = True, show = True, log10 = True, ylim = None, xlim = None):
    plt.clf()
    real_eigs = pd.DataFrame({h11:real_eigs})[h11]

    #histogram(real_eig_df[10], '')

    data_tensor = gen_matrices(GAN, args, noise=noise)
    #data_tensor = batch_upper_left(data_tensor, h11)
    eigs = ensemble_eigenvalues(data_tensor)
    eigs = [k for k in eigs if k not in [np.nan, np.inf, -np.inf]]
    wishart_eigs = ensemble_eigenvalues(gen_wishart_matrices(h11,batchSize))
    
    log_eigs, log_real_eigs = [k for k in np.log10(eigs) if not np.isnan(k) and k not in [np.nan, np.inf, -np.inf]], [k for k in np.log10(np.array(real_eigs)) if not np.isnan(k) and k not in [np.nan, np.inf, -np.inf]]
    if display_wishart:
        log_wishart_eigs = [k for k in np.log10(wishart_eigs) if not np.isnan(k) and  k not in [np.nan, np.inf, -np.inf]]

    nologdist, logdist = wasserstein_distance(eigs,real_eigs), wasserstein_distance(log_eigs, log_real_eigs)
    if log10:
        dist = logdist
    else:
        dist = nologdist

    title = r'$h^{11} = ' + str(h11) + ',\,\,\, n_z = ' + str(nz) +",\,\,\, \mathrm{\,\,\, Distance = }" + str(round(dist,2)) + "$"
    xlabel = r'$\mathrm{log}_{10}(\mathrm{Eigenvalue})$'

    histogram(real_eigs, title=title, show = False, log10 = log10, norm_hist = norm_hist)
    if display_wishart:
        histogram(pd.DataFrame({h11: wishart_eigs})[h11], title=title, kde = kde, norm_hist = norm_hist, show = False, log10 = log10)
    histogram(pd.DataFrame({h11: eigs})[h11], title=title, kde = kde, norm_hist = norm_hist, show = show, log10 = log10, ylim=ylim, xlim=xlim, \
             filename = args.outdir + '/h11_' + str(h11) + '_nz_' +str(nz)+'_epoch_'+str(epoch)+'_plot.png', xlabel = xlabel, dpi=dpi)
    
#     return wasserstein_distance(eigs,real_eigs), wasserstein_distance(wishart_eigs,real_eigs), wasserstein_distance(log_eigs,log_real_eigs)
