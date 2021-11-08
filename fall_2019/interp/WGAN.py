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

def get_fake_labels(vals,num_each,length,shuffle=True):
    all_fakes = []
    for idx in range(len(vals)):
        val, num = vals[idx], num_each[idx]
        all_fakes.extend([[1.0 if k == val - 1 else 0.0 for k in range(length)] for _ in range(num)])
    return torch.tensor(all_fakes)

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
                _ , _, netG = WGAN_list[-1]
                wass, wass_log, fake_eigs, log_fake_eigs = {}, {}, {}, {}
                for h11 in real_eigs:
                    wass[h11], wass_log[h11], fake_eigs[h11], log_fake_eigs[h11] = test_generator(netG,h11,args,real_eigs[h11])

#                wass, wass_log = test_generator(netG, args, real_eigs)
                print('\n\nEpoch [%d/%d], Wasserstein(eigs,real_eigs): %s, Wasserstein(log_eigs,log_real_eigs): %s'
                      % (epoch,
                         args.epochs,
                         #critic_loss.data,
                         #generator_loss.data,
                         #err_real.data.mean(),
                         #err_fake.data.mean(),
                         str(wass),
                         str(wass_log))
                      )

                print("\n\t\tChecking spread of fake distributions:")
                keys = list(fake_eigs.keys())
                for i in range(len(keys)):
                    for j in range(i+1,len(keys)):
                        k1, k2 = keys[i], keys[j]
                        r1, r2 = fake_eigs[k1], fake_eigs[k2]
                        l1, l2 = log_fake_eigs[k1], log_fake_eigs[k2]
                        print("\t\t\t(h11_1,h11_2) = (%d,%d): %.3f, %.3f" % (k1,k2,wasserstein_distance(r1,r2),wasserstein_distance(l1,l2)))



                # if epoch % args.plot_interval == 0: show_GAN_histogram(netG, epoch, h11, args, real_eigs,
                #                    batchSize=args.num_geometries, nz=args.nz, log10=True, dpi=300, display_wishart=False, ylim=(0, 1),
                #                    xlim=(-6, 2), show=args.show_plots)

            # Keep track of critic step
            crit_steps = 0

            for batch_number, (data,labels) in enumerate(data_loader):
                images = data
                batch_size = images.shape[0]
                images = to_var(images.view(batch_size, -1))
                labels = to_var(labels)

                # 1) Train critic
                if crit_steps < args.n_critic_steps:
                    critic.zero_grad()
                    generator.zero_grad()

                    # clamp to cube
                    for p in critic.parameters():
                        p.data.clamp_(-CLAMP, CLAMP)

                    # compute error on real images
                    #print(images.shape)
                    critic_real = critic(images,labels)
                    err_real = torch.mean(critic_real)
                    err_real = err_real.view(1)

                    # draw random vars, generate fakes, and compute fake errors
                    z = to_var(torch.randn(batch_size, args.nz))
                    num_each = [batch_size//len(args.h11s_test) for _ in args.h11s_test[:-1]]
                    num_each.append(batch_size-sum(num_each))
                    fake_labels = to_var(get_fake_labels(args.h11s_test,num_each,args.max_h11))
                    fake_images = generator(z,fake_labels)
                    err_fake = torch.mean(critic(fake_images,fake_labels))

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
                    num_each = [batch_size//len(args.h11s_test) for _ in args.h11s_test[:-1]]
                    num_each.append(batch_size-sum(num_each))
                    fake_labels = to_var(get_fake_labels(args.h11s_test,num_each,args.max_h11))
                    fake_images = generator(z,fake_labels)
                    outputs = critic(fake_images,fake_labels)

                    # Minimize this so we max its negative
                    generator_loss = -torch.mean(outputs)

                    # Backprop and optimizer generator
                    generator_loss.backward()
                    generator_optimizer.step()

                    # Reset the counter so the critic gets the next batch
                    crit_steps = 0


    except KeyboardInterrupt:
        print("Training ended via keyboard interrupt.")