import sys
import caffe
import os
from copy import copy,deepcopy
sys.path.append('../code')

from scipy import ndimage
from argparse import ArgumentParser
from numpy import mean, ceil, std, inf, sqrt, hstack
from numpy.random import rand
from scipy.io import loadmat
from tools import Experiment, mapp
import numpy as np
from ride.slstm import SLSTM
from collections import defaultdict
from cmt.transforms import WhiteningPreconditioner
from cmt.tools import generate_data_from_image
from skimage.util.shape import view_as_windows
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import scipy.stats 
from matplotlib import cm
from skimage.restoration import denoise_bilateral
from skimage import feature
from skimage.filters import sobel
from scipy.stats import norm
from scipy.special import expit
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
from copy import copy
from skimage.measure import compare_ssim as ssim, compare_psnr as psnr
from skimage.color import rgb2gray
from scipy.linalg import qr


def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--model',        '-m', type=str, default='models/1_layer_ride/rim.10082016.221222.xpck')
	parser.add_argument('--data',       '-d', type=str, default='data/BSDS_Cropped/img_data.mat')
	parser.add_argument('--noise_std',  '-n', type=int, default=-1)
	parser.add_argument('--momentum',	'-M', type=float, default=0.9)
	parser.add_argument('--lr',  		'-l', type=float, default=5.0)
	parser.add_argument('--niter',  	'-N', type=int, default=400)
	parser.add_argument('--path',       '-p', type=str, default='map_single_pixel/')
	parser.add_argument('--mode',       '-q', type=str, default='CPU', choices=['CPU', 'GPU'])
	parser.add_argument('--device',     '-D', type=int, default=0)
	parser.add_argument('--size',		'-s', type=int, default= 160)
	parser.add_argument('--samples', '-y', type=float,	  default = 0.4)
	parser.add_argument('--image_num', '-K', type=int,  default = 2)
	parser.add_argument('--resume',	'-r',	type=int,default=-1)
	parser.add_argument('--flip',	'-f', type=int,default=0)	
	parser.add_argument('--ent_max', '-e', type=float, default=100.0)

	args = parser.parse_args(argv[1:])
	niter = args.niter
	lr = args.lr
	N = args.size
	K = args.image_num
	noise_std = args.noise_std
	path = args.path
	if not os.path.exists(path):
		os.makedirs(path)	
		
	print 'Measurement Rate',args.samples
	print 'Noise Level',noise_std
	# select CPU or GPU for caffe
	if args.mode.upper() == 'GPU':
		print "setting the GPU mode"
		caffe.set_mode_gpu()
		caffe.set_device(args.device)
	else:
		caffe.set_mode_cpu()
	
	if noise_std>-1:
		noise_std = float(noise_std)*10/255.0
		path+=str(args.noise_std)+'/'
	if args.samples > -1:
		path += str(args.samples)+'/'

	if not os.path.exists(path):
		os.makedirs(path)	
	sys.stdout = open(path+'log.txt','w')
	# load data
	if args.data.lower()[-4:] in ['.gif', '.png', '.jpg', 'jpeg']:
	    images = plt.imread(args.data)

	    img = rgb2gray(images[:args.size,:args.size])
	    img = img.astype('float64')
	else:
		images = loadmat(args.data)['data']
		images = images.astype('float64')
		img = images[:K,:args.size,:args.size]

	# load model
	print 'Loading model'
	experiment = Experiment(args.model)
	model = experiment['model']
	input_mask = model.input_mask
	output_mask = model.output_mask

	Phi = np.load('map_single_pixel/Phi_g_'+str(N)+'.npy')[1:int(args.samples*args.size**2),:]
	del images


	for k in range(K):
		if not os.path.exists(path+str(k)+'/'):
			os.makedirs(path+str(k)+'/')
		mplimg.imsave(path+str(k)+'/original_img',img[k].squeeze(),cmap=cm.gray)	
	
	y = np.dot(Phi,img.reshape(K,-1).transpose())	
	np.random.seed(123)
	if noise_std>-1:
		y += noise_std*np.random.randn(*y.shape)
	M = y.shape[0]
	print 'Number of measurements', M

	#Initializing
	np.random.seed(123)
	init_img = np.random.rand(N**2,K)

	prev_grad = 0
	if args.resume > 0:
		init_img = np.load(path+'cleaned_img/'+str(args.resume)+'.npy')

	for k in range(K):
		mplimg.imsave(path+str(k)+'/init_img', init_img[:,k].reshape(N,N), cmap = cm.gray)
	
	psnr_list = [[]for i in range(K)]
	ssim_list = [[]for i in range(K)]
	for i in range(args.niter):

		j = args.flip*i
		f,grad_img,whitened_img = model.gradient(init_img.transpose().reshape(K,N,N,1)[:,::(-1)**j,::(-1)**(j/2),:],precond = None,niter=i,path=path,ent_max=args.ent_max)
		df_dh = grad_img[:,::(-1)**j,::(-1)**(j/2),:].reshape(K,-1).transpose()
		print i, 'f',f.sum(),'df_dh', np.abs(df_dh).sum(),
		prev_grad = args.momentum*prev_grad + df_dh
		x_up = init_img+ lr*(prev_grad)

		init_img = x_up - np.dot(Phi.transpose(),np.dot(Phi,x_up)-y)
		init_img=np.clip(init_img,0.0,1.0)

		if i%10 == 0:
			for k in range(K):
				mplimg.imsave(path+str(k)+'/img'+str(i), init_img[:,k].reshape(N,N), cmap = cm.gray)
		l = linalg.norm(y-np.dot(Phi,init_img))
		print 'l',l

		#For Saving Gradient Image

		# if (i%10==0):
		# 	for k in range(K):
		# 		fig2 = plt.figure(2)
		# 		plt.imshow(df_dh[:,k].reshape(N,N),vmin = -0.02,vmax=0.02)
		# 		plt.colorbar()
		# 		plt.savefig(path+str(k)+'/grad_img'+str(i))
		# 		plt.close(fig2)	

		# if (i%1 ==0):
		# 	for k in range(K):
		# 		fig1 = plt.figure(1)
		# 		plt.imshow(f[:,k].reshape(N,N))
		# 		plt.colorbar()
		# 		plt.savefig(path+str(k)+'/loglik_img'+str(i))
		# 		plt.close(fig1)


		m= 2 #Margin to remove for comparision
		for k in range(K):
			ssim1 = ssim(init_img[:,k].reshape(N,N)[m:-m,m:-m],img[k].squeeze()[m:-m,m:-m],dynamic_range=img.min()-img.max())
			psnr1 = psnr(init_img[:,k].reshape(N,N)[m:-m,m:-m],img[k].squeeze()[m:-m,m:-m],dynamic_range=img.min()-img.max())
			ssim_list[k].append(ssim1)
			psnr_list[k].append(psnr1)
			if not os.path.exists(path+'cleaned_img/'):
				os.makedirs(path+'cleaned_img/')	
			np.save(path+'cleaned_img/ssim_list',ssim_list)
			np.save(path+'cleaned_img/psnr_list',psnr_list)
			print k,'ssim',ssim1,'psnr', psnr1

		if (i%50==0): #Storing npy files
			np.save(path+'cleaned_img/'+str(i),init_img)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
