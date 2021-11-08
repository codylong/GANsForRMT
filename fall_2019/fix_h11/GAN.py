from generator import *
from analysis import *
from torch.autograd import Variable
import copy

def to_var(x):
    # first move to GPU, if necessary
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def train_GAN(args, G, D, data_loader, real_eigs, num_test_geometries):

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()

    # Define your loss function and optimizer by picking from the ones already available in `torch.nn` or defining your own
    criterion = nn.BCELoss()  # Or remove nn.Sigmoid() and use
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr_D)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr_G)

    GAN_list = []

    try:
        for epoch in range(args.epochs + 1):

            ###
            # Print progress as training occurs
            ###

            if (epoch) % args.log_interval == 0:
                GAN_list.append((args.h11, epoch, copy.deepcopy(G).cpu()))
                h11, _, netG = GAN_list[-1]
                wass, wass_log = test_generator(netG, args, real_eigs)
                print('Epoch [%d/%d], Wasserstein(eigs,real_eigs): %.3f, Wasserstein(log_eigs,log_real_eigs): %.3f'
                      % (epoch,
                         args.epochs,
                         #d_loss.data,
                         #g_loss.data,
                         #real_score.data.mean(),
                         #fake_score.data.mean(),
                         wass,
                         wass_log)
                      )

                if epoch % args.plot_interval == 0: show_GAN_histogram(netG, epoch, h11, args, real_eigs,
                                   batchSize=args.num_geometries, nz=args.nz, log10=True, dpi=300,
                                   display_wishart=False, ylim=(0, 1),
                                   xlim=(-6, 2), show=args.show_plots)

            for batch_number, data in enumerate(data_loader):

                images = data[0]
                batch_size = images.shape[0]  # b/c could be < batchSize for last batch

                images = to_var(images.view(batch_size, -1))

                ###
                # TRAIN DISCRIMINATOR
                ###

                # Create discriminator targets
                real_labels = to_var(torch.ones(batch_size, 1))
                fake_labels = to_var(torch.zeros(batch_size, 1))

                # Pass through discriminator, record score, compute loss
                outputs = D(images)
                real_score = outputs
                d_loss_real = criterion(outputs, real_labels)

                # Generate images from noise
                z = to_var(torch.randn(batch_size, args.nz))
                fake_images = G(z)

                # Evaluate discriminator on fake images, record score, compute loss
                outputs = D(fake_images)
                fake_score = outputs
                d_loss_fake = criterion(outputs, fake_labels)

                # Backprop and optimize discriminator
                d_loss = d_loss_real + d_loss_fake  # total loss is sum of parts
                D.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                ###
                # TRAIN GENERATOR
                ###

                for g_epoch in range(5): # five generator epochs per discriminator epoch
                    # Generate images from noise
                    z = to_var(torch.randn(batch_size, args.nz))
                    fake_images = G(z)
                    outputs = D(fake_images)

                    # Compute G loss: G does well if D thinks fake is real
                    g_loss = criterion(outputs, real_labels)

                    # Backprop and optimize generator
                    D.zero_grad()
                    G.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()

    except KeyboardInterrupt:
        print("Training ended via keyboard interrupt.")