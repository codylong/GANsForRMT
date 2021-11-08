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
from VAE import VAE


parser = argparse.ArgumentParser(description='VAE KS Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset', help='cifar10 | lsun | imagenet | folder | lfw | h11_10 | h11_20 | h11_30')
parser.add_argument('--numgeometries', type=int, default=2000, help='number of geometries')
parser.add_argument('--dataroot', type=str, help='path to dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
if args.dataset == 'h11_10':
    h11 = 10
elif args.dataset == 'h11_20':
    h11 = 20
elif args.dataset == 'h11_30':
    h11 = 30

num_pixels = h11*h11

if 'h11' in args.dataset:
    num_geometries = args.numgeometries
    newsize = h11
    diff = newsize-h11
    files_list = [d for d in os.listdir(args.dataroot) if "_"+str(h11)+"_" in d and 'K' in d]
    print("TOTAL POSSIBLE GEOMS", len(files_list))
    files_list = files_list[:num_geometries]
    print(args.dataroot,type(args.dataroot),files_list[0],type(files_list[0]))
    print("First five files,", files_list[:5])
    raw_pickles = []
    for d in files_list:
        try:
            raw_pickles.append(pickle.load(open(args.dataroot + d,'rb')))#,encoding='latin1'))
        except ValueError:
            print("Error reading pickle:",d)
    padded_raw_pickles = [torch.nn.functional.pad(transforms.ToTensor()(transforms.ToPILImage()(torch.Tensor(k))),(0,diff,0,diff),'constant',0).numpy() for k in raw_pickles]
    
    num_geometries = len(padded_raw_pickles)
    print("After loading, number of geometries:",num_geometries)
    raw_data = torch.tensor(padded_raw_pickles).view(num_geometries,1,newsize,newsize)
    dataset = torch.utils.data.TensorDataset(raw_data)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                        shuffle=True)

model = VAE(h11).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #BCE = F.l1_loss(recon_x, x.view(-1, num_pixels))
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, num_pixels), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if epoch % 1000 == 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data[0]
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, h11, h11)[:n]])
                save_image(comparison.cpu(),
                         'results_ks/reconstruction_' + str(epoch) + '.png', nrow=n)
 
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(0, args.epochs+1):
        train(epoch)
        if epoch % 1000 == 0:
            test(epoch)
            with torch.no_grad():
                torch.save(model.state_dict(),'results_ks/model_' + str(epoch) + '.pth')

                sample = torch.randn(64, 20).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, h11, h11),
                       'results_ks/sample_' + str(epoch) + '.png')
