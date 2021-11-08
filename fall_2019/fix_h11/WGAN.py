from generator import *
from analysis import *
from torch.autograd import Variable
from matplotlib import pyplot as plt
import copy

def to_var(x):
    # first move to GPU, if necessary
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def train_WGAN(args, generator, critic, data_loader, real_eigs, num_test_geometries):
    CLAMP = 0.01

    if torch.cuda.is_available():
        critic.cuda()
        generator.cuda()

    critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=args.lr_D)
    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.lr_G)

    WGAN_list = []

    try:
        for epoch in range(args.epochs + 1):

            ###
            # Print progress as training occurs
            ###

            if (epoch) % args.log_interval == 0:
                WGAN_list.append((args.h11, epoch, copy.deepcopy(generator).cpu()))
                h11, _, netG = WGAN_list[-1]
                wass, wass_log = test_generator(netG, args, real_eigs)
                print('Epoch [%d/%d], Wasserstein(eigs,real_eigs): %.3f, Wasserstein(log_eigs,log_real_eigs): %.3f'
                      % (epoch,
                         args.epochs,
                         #critic_loss.data,
                         #generator_loss.data,
                         #err_real.data.mean(),
                         #err_fake.data.mean(),
                         wass,
                         wass_log)
                      )

                if epoch % args.plot_interval == 0: show_GAN_histogram(netG, epoch, h11, args, real_eigs,
                                   batchSize=args.num_geometries, nz=args.nz, log10=True, dpi=300, display_wishart=False, ylim=(0, 1),
                                   xlim=(-6, 2), show=args.show_plots)

            # Keep track of critic step
            crit_steps = 0

            for batch_number, data in enumerate(data_loader):
                images = data[0]
                batch_size = images.shape[0]
                images = to_var(images.view(batch_size, -1))

                # 1) Train critic
                if crit_steps < args.n_critic_steps:
                    critic.zero_grad()
                    generator.zero_grad()

                    # clamp to cube
                    for p in critic.parameters():
                        p.data.clamp_(-CLAMP, CLAMP)

                    # compute error on real images
                    critic_real = critic(images)
                    err_real = torch.mean(critic_real)
                    err_real = err_real.view(1)

                    # draw random vars, generate fakes, and compute fake errors
                    z = to_var(torch.randn(batch_size, args.nz))
                    fake_images = generator(z)
                    err_fake = torch.mean(critic(fake_images))

                    # Minimize this means maximize err_real - err_fake
                    critic_loss = err_fake - err_real
                    critic_loss.backward()
                    critic_optimizer.step()

                    # updated critic, remember that!
                    crit_steps += 1
                # 2) Train generator
                else:
                    critic.zero_grad()
                    generator.zero_grad()

                    # draw random vars, generate fakes, and compute fake errors
                    z = to_var(torch.randn(batch_size, args.nz))
                    fake_images = generator(z)
                    outputs = critic(fake_images)

                    # Minimize this so we max its negative
                    generator_loss = -torch.mean(outputs)

                    # Backprop and optimizer generator
                    generator_loss.backward()
                    generator_optimizer.step()

                    # Reset the counter so the critic gets the next batch
                    crit_steps = 0


    except KeyboardInterrupt:
        print("Training ended via keyboard interrupt.")