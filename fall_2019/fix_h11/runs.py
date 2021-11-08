import subprocess

# models = ['WGAN_DCGAN', 'WGAN_FFNN', 'GAN_DCGAN', 'GAN_FFNN']
# h11s = [10,20,30,40,50]
# nzs = [5, 15, 25, 50]
# lrs = [.00005, .000005]

models = ['WGAN_DCGAN']
h11s = [50]
nzs = [200]
lrs = [.00005, .000005]


for model in models:
	for h11 in h11s:
		for nz in nzs:
			for lr in lrs:
				command = "python main.py --save --model-type %s --h11 %d --nz %d --lr-G %.6f --lr-D %.6f" % (model,h11,nz,lr,lr)
				process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
				process.wait()
				print(command)