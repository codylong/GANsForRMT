import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN_D(nn.Module):
    #def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
    def __init__(self, h11, nz, nc=1, ndf=64, ngpu=1, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        
        isize = h11
        #isize, nz = args.h11, args.nz
        self.ngpu = ngpu
        #assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.h11 = h11
        self.nz = nz

        sizes = [16,32,64,128]
        for this_size in sizes:
            if this_size >= self.h11: break

        main = nn.Sequential()

        if this_size != self.h11:
            isize = this_size
            main.add_module('scale_up:{0}-{1}:linear'.format(self.h11,isize),
                    nn.Linear(self.h11*self.h11,isize*isize))
            main.add_module('scale_up:{0}:relu'.format(isize),
                    nn.LeakyReLU(0.2, inplace=True))
           
        self.isize = isize

        print("data:", isize, self.h11)

        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf
        
        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        s = input.shape
        input = input.view(s[0],1,self.h11,self.h11)

        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            if self.h11 != self.isize:
                output = input
                s = output.shape
                output = output.view(s[0],s[1],self.h11*self.h11)
                output = self.main[0](output)
                output = output.view(s[0],s[1],self.isize,self.isize)
                for push in self.main[1:]:
                    output = push(output)
            else:
                output = self.main(input)
                
        return output.view(output.shape[0],output.shape[1])


class DCGAN_D_Sigmoid(DCGAN_D):
    # def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
    def __init__(self, h11, nz, nc=1, ndf=64, ngpu=1, n_extra_layers=0):
        super(DCGAN_D_Sigmoid, self).__init__(h11, nz, nc=1, ndf=64, ngpu=1, n_extra_layers=0)
        self.main.add_module('Final sigmoid:', nn.Sigmoid())

    def forward(self, input):
        s = input.shape
        input = input.view(s[0], 1, self.h11, self.h11)

        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            if self.h11 != self.isize:
                output = input
                s = output.shape
                output = output.view(s[0], s[1], self.h11 * self.h11)
                output = self.main[0](output)
                output = output.view(s[0], s[1], self.isize, self.isize)
                for push in self.main[1:]:
                    output = push(output)
            else:
                output = self.main(input)

        # output = output.mean(0)
        # output = output.view(1)
        return output.view(output.shape[0],output.shape[1])


if __name__ == '__main__':
    #h11, nz = 33, 50
    nz, nc = 5, 1
    for h11 in range(62,66):
        model = DCGAN_D(h11,nz,nc=nc)
        images = torch.randn(100,nc,h11,h11)
        print(model(images).shape)