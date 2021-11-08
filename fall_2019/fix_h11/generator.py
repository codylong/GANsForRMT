from torch import nn
import torch

class symmetric_G_FFNN(nn.Module):
    def __init__(self, args):
        super(symmetric_G_FFNN, self).__init__()
        self.h11 = args.h11
        self.nz = args.nz
        self.fc1, self.fc2, self.fc3, self.fc4, self.fc5 = nn.Linear(self.nz, 256), nn.Linear(256, 256), nn.Linear(256,256), nn.Linear(256, 256), nn.Linear(256, self.h11 * self.h11)
        self.a1, self.a2, self.a3, self.a4 = nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), nn.LeakyReLU(0.2)

    def forward(self, input):
        o1 = self.a1(self.fc1(input))
        o2 = self.a1(self.fc2(o1))
        o3 = self.a1(self.fc3(o2))
        o4 = self.a1(self.fc4(o3))
        o5 = self.fc5(o4)
        o, oT, s = o5.view(o5.shape[0], self.h11, self.h11), torch.transpose(o5.view(o5.shape[0], self.h11, self.h11),
                                                                             1, 2), o5.shape
        m = torch.bmm(o, oT)
        return m.view(m.shape[0], self.h11 * self.h11)


class DCGAN_G(nn.Module):
    #def __init__(self, args, nc=1, ngf=64, ngpu=1, n_extra_layers=0):
    def __init__(self, h11, nz, nc=1, ngf=64, ngpu=1, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        isize = h11
        #isize, nz = args.h11, args.nz
        self.ngpu = ngpu
        #assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.h11 = h11
        self.nz = nz

        sizes = [16,32,64,128]
        for this_size in sizes:
            if this_size > self.h11: break

        if this_size != self.h11:
            isize = this_size

        self.isize = isize

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        

        if isize != h11:
            main.add_module("Down to H11", nn.Linear(isize*isize,h11*h11))

        main.add_module('final:{0}:tanh'.format(nc), nn.Sequential())
        self.main = main

    def forward(self, input):
        input = input.view(input.shape[0],input.shape[1],1,1)

        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = input

            if self.h11 != self.isize:
                for push in self.main[:len(self.main)-2]:
                    output = push(output)
                s = output.shape
                to_h11, last = self.main[-2], self.main[-1]
                output = to_h11(output.view(s[0],s[1],s[2]*s[3]))
                output = last(output)
                output = output.view(s[0],s[1], self.h11, self.h11)
            else:
                for push in self.main: output = push(output)
                
        #return output
        o, oT, s = output, torch.transpose(output, 2, 3), output.shape
        m = torch.bmm(o.view(s[0] * s[1], s[2], s[3]), oT.view(s[0] * s[1], s[2], s[3])).view(s[0], s[1], s[2], s[3])
        return m.view(m.shape[0], self.h11 * self.h11)

if __name__ == '__main__':
    #h11, nz = 33, 50
    nz = 5
    for h11 in range(9,12):
        model = DCGAN_G(h11,nz,nc=1)
        images = torch.randn(100,nz,1,1)#,h11,h11)
        print(model(images).shape)
