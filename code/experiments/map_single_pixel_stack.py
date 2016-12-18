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

	print 'Measurement Rate',args.samples
	print 'Noise Level',noise_std
	# select CPU or GPU for caffe
	if args.mode.upper() == 'GPU':
		print "setting the GPU mode"
		caffe.set_mode_gpu()
		caffe.set_device(args.device)
	else:
		caffe.set_mode_cpu()
	path = args.path
	
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
	    print 'img max',img.max()
	else:
		images = loadmat(args.data)['data']
		images = images.astype('float64')
			# load an image, 56 looks good
		# idxs = [2,4]
		# images = images[idxs,:,:]
		img = images[:K,:args.size,:args.size]
		print 'shape', images.shape
		print 'img max',img.max()

	# load model
	print 'Loading model'
	experiment = Experiment(args.model)
	model = experiment['model']
	#print 'precond',model.preconditioner.logjacobian
	input_mask = model.input_mask
	output_mask = model.output_mask

	Phi = loadmat('map_single_pixel/Phi_g_'+str(N)+'.mat')['Phi'][1:int(args.samples*args.size**2),:]
	# y  = loadmat(path+'random/Ball_64.mat')['y'][1:int(args.samples*args.size**2),:]
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
	print 'Starting pinv'
	# init_img = np.linalg.lstsq(Phi,y)
	# init_img = init_img[0]
	np.random.seed(123)
	init_img = np.random.rand(N**2,K)
	# init_img = img.reshape(N**2,1)
	# init_img = np.zeros_like(init_img)
	# print 'init_img min', init_img.min()
	# print 'init_img max', init_img.max()	
	# init_img = init_img/init_img.max()
	# init_img[init_img<0]= 0.0
	# plt.imshow(init_img, cmap = cm.gray)
	# plt.colorbar()
	# plt.show()
	# init_img = np.zeros((N,N))
	# i_max = copy(init_img.max())
	# i_min = copy(init_img.min())
	# i_min = 0
	prev_grad = 0
	if args.resume > 0:
		init_img = np.load(path+'cleaned_img/'+str(args.resume)+'.npy')
		# init_img = np.load('/home/cplab-ws1/ride/code/map_single_pixel/0.3/cleaned_img/200.npy')

	for k in range(K):
		mplimg.imsave(path+str(k)+'/init_img', init_img[:,k].reshape(N,N), cmap = cm.gray)
	
	psnr_list = [[]for i in range(K)]
	ssim_list = [[]for i in range(K)]
	for i in range(args.niter):

		# if i%300 == 0:
		# 	lr = 0.8
		j = args.flip*i
		f,grad_img,whitened_img = model.gradient(init_img.transpose().reshape(K,N,N,1)[:,::(-1)**j,::(-1)**(j/2),:],precond = None,niter=i,path=path,ent_max=args.ent_max)
		# print ((init_img-i_min)/(i_max-i_min)).max()
		# print ((init_img-i_min)/(i_max-i_min)).min()
		df_dh = grad_img[:,::(-1)**j,::(-1)**(j/2),:].reshape(K,-1).transpose()
		# dl_dh = -2*np.dot(Phi.transpose(),y-np.dot(Phi,init_img.reshape(N**2,1))).reshape(N,N)/N**2
		print i, 'f',f.sum(),'df_dh', np.abs(df_dh).sum(),
		prev_grad = args.momentum*prev_grad + df_dh
		x_up = init_img+ lr*(prev_grad)
		# if entropy.max()>5.0:
		# 	print  'Sampling'
		# 	x_up_mean = np.zeros((N**2,1)) 
		# 	for it in range(1):
		# 		x,x_sample,loglik_img = model.interpolate(x_up.reshape(1,N,N,1),entropy<3.8,method='mode',sample=1)
		# 		x_up_mean +=x_sample.reshape(N**2,1)
		# 	x_up_mean = x_up_mean/1

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

		# if (i%1==0):
		# 	for k in range(K):
		# 		fig3 = plt.figure(3)
		# 		plt.imshow(whitened_img[:,k].reshape(N,N),vmin=-5.0,vmax=6.0)
		# 		plt.colorbar()
		# 		plt.savefig(path+str(k)+'/whitened_img'+str(i))
		# 		plt.close(fig3)	

		# if i%1==0 and entropy.max()>4.0:
		# 	fig3 = plt.figure(4)
		# 	plt.imshow(x_sample.reshape(N,N),cmap='gray')
		# 	plt.colorbar()
		# 	plt.savefig(path_sample'+str(i))
		# 	plt.close(fig3)	

		m= 2
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

		if (i%50==0):
			np.save(path+'cleaned_img/'+str(i),init_img)

	model.verbosity=0


	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
