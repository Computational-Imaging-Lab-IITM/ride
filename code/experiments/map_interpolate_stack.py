import sys
import caffe
from copy import copy
import os
sys.path.append('../code')

from scipy import ndimage
from argparse import ArgumentParser
from numpy import mean, ceil, std, inf, sqrt, hstack
from numpy.random import rand
from scipy.io import loadmat
from tools import Experiment, mapp
import numpy as np
np.random.seed(123)
from ride.slstm import SLSTM
from collections import defaultdict
from cmt.transforms import WhiteningPreconditioner
from cmt.tools import generate_data_from_image
import numpy as np
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import scipy.stats 
from matplotlib import cm
from skimage.measure import compare_ssim as ssim, compare_psnr as psnr
from skimage.color import rgb2gray
from tools.utils import fd


def get_mask(img, margin, holes):
	np.random.seed(123)
	mask = np.random.choice(2,img.shape,p=[holes,1-holes])
	# mask[:margin,:] = 1
	# mask[:,:margin] = 1
	# mask[:,-margin:] = 1
	# mask[-margin:,:] = 1
	return mask

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--model',        '-m', type=str, default='models/1_layer_ride/rim.10082016.221222.xpck')
	parser.add_argument('--data',       '-d', type=str, default='data/BSDS_Cropped/img_data.mat')
	parser.add_argument('--holes',	    '-H', type=float, default=0.7)
	parser.add_argument('--momentum',	'-M', type=float, default=0.9)
	parser.add_argument('--lr',  		'-l', type=float, default=5.0)
	parser.add_argument('--niter',  	'-N', type=int, default=10000)
	parser.add_argument('--path',       '-p', type=str, default='/home/cplab-ws1/ride/code/map_interpolate/')
	parser.add_argument('--mode',       '-q', type=str,   default='CPU', choices=['CPU', 'GPU'])
	parser.add_argument('--device',     '-D', type=int,   default=0)
	parser.add_argument('--size',		'-s', type=int, default= 256)
	parser.add_argument('--flip',	'-f', type=int,default=1)
	parser.add_argument('--ent_max', '-e', type=float, default=3.5)
	parser.add_argument('--resume','-r',type=int,default= 0)
	parser.add_argument('--index','-I',type=int,default=0)
	
	args = parser.parse_args(argv[1:])
	niter = args.niter
	lr = args.lr
	N = args.size
	path = args.path
	if not os.path.exists(path):
		os.makedirs(path)		

	# select CPU or GPU for caffe
	if args.mode.upper() == 'GPU':
		print "setting the GPU mode"
		caffe.set_mode_gpu()
		caffe.set_device(args.device)
	else:
		caffe.set_mode_cpu()


	if args.index > -1:
		path += str(args.index)+'/'
	print path
	if not os.path.exists(path):
		os.makedirs(path)	
	sys.stdout = open(path+'log'+str(args.index)+'.txt','w')
	print 'in log'


	# load data
	if args.data.lower()[-4:] in ['.gif', '.png', '.jpg', 'jpeg']:
	    images = plt.imread(args.data)
	    img = rgb2gray(images[:args.size,:args.size])
	    print 'img max',img.max()
	    img = img.astype('float64')
	    #img = (img/255.0)  
	    #vmin, vmax = 0, 255

	else:
		images = loadmat(args.data)['data']
		img = images[args.index,:args.size,:args.size].squeeze()
		if img.max() > 2:
			img.astype('float64')
			img = img/255.0
		print 'img shape', img.shape
		del images

	
	# load model
	experiment = Experiment(args.model)
	model = experiment['model']
	input_mask = model.input_mask
	output_mask = model.output_mask

	mplimg.imsave(path+'original_img',img,cmap=cm.gray)
	binary_mask = get_mask(img,0, holes = args.holes)
	mplimg.imsave(path+'noisy_img',binary_mask*img,cmap=cm.gray)

	sizer = 130
	sizec = 140
	images = []
	stacked_masks = []
	for i in [0,126]:
		for j in [0,116]:
			images.append(img[i:i+sizer,j:j+sizec])
			stacked_masks.append(binary_mask[i:i+sizer, j:j+sizec])

	images = np.expand_dims(np.array(images),axis=4)
	stacked_masks = np.expand_dims(np.array(stacked_masks),axis=4)
	print images.shape, stacked_masks.shape	

	masked_images = images*stacked_masks

	if args.resume == 0:
		np.random.seed(10)
		init_img = masked_images + (1-stacked_masks)*(np.random.rand(*images.shape))

	else :
		init_img = np.load(path+'cleaned_img/'+str(args.resume)+'.npy')

	print init_img.min()	
	print init_img.max()	
	prev_update = 0
	print 'init shape', init_img.shape
	for i in range(args.niter):
		# if i%50==0 and i!= 0:
		# 	lr = lr/2.0
		j = args.flip*i # To enable flipping directions	
		f, grad_img, whitened_img = model.gradient(init_img[:,::(-1)**j,::(-1)**(j/2),:],path=path,niter=i,ent_max = args.ent_max)

		f = f.sum()
		grad_img = grad_img[:,::(-1)**j,::(-1)**(j/2),:]
		whitened_img = whitened_img[:,::(-1)**j,::(-1)**(j/2),:]

		df_dh = lr*grad_img
		print i, 'loglik', f,'df_dh', (df_dh).sum()

		current_update = lr*grad_img + args.momentum*prev_update
		init_img += (1-stacked_masks)*current_update 
		init_img = np.clip(init_img,0.0,1.0)
		prev_update = current_update
	
		#Printing results
		if (i%1==0):
			for k in range(images.shape[0]):
				m = 4
				cleaned_img =init_img[k,m:,m:-m,:].squeeze()
				img_k = images[k,m:,m:-m,:].squeeze()		
				if not os.path.exists(path+str(k)+'/'):
					os.makedirs(path+str(k)+'/')		
				mplimg.imsave(path+''+str(k)+'/cleaned_img'+str(i), cleaned_img, cmap = cm.gray,vmin=0,vmax=1)
				ssim1 = ssim(cleaned_img,img_k,dynamic_range=img_k.max()-img_k.min())
				psnr1 = psnr(cleaned_img,img_k,dynamic_range=img_k.max()-img_k.min())
				print k, 'ssim',ssim1,'psnr', psnr1 #, 'min',cleaned_img.min(),'max',cleaned_img.max()

		# if (i%10==0):
		# 	for k in range(images.shape[0]):
		# 		fig2 = plt.figure(2)
		# 		plt.imshow(grad_img[k,:,:,:].squeeze(),vmin = -0.02,vmax=0.02)
		# 		plt.colorbar()
		# 		plt.savefig(path+''+str(k)+'/grad_img'+str(i))
		# 		plt.close(fig2)	

		# 		fig3 = plt.figure(3)
		# 		plt.imshow(whitened_img[k,:,:,:].squeeze(),vmin=-5.0,vmax=6.0)
		# 		plt.colorbar()
		# 		plt.savefig(path+''+str(k)+'/whitened_img'+str(i))
		# 		plt.close(fig3)	

		if (i%20==0):
			if not os.path.exists(path+'cleaned_img/'):
				os.makedirs(path+'cleaned_img/')	
			np.save(path+'cleaned_img/'+str(i),init_img)

	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
