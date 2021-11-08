import torch
from torch import nn
from torch.nn import functional as F

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
        h4 = torch.sigmoid(self.fc4(h3))
        h4 = h4.view(h4.shape[0],self.h11,self.h11)
        o, oT, s = h4, torch.transpose(h4,1,2), h4.shape
        h4 = torch.bmm(o,oT)/self.h11
        return h4.view(s[0],self.h11*self.h11)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.h11*self.h11))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
