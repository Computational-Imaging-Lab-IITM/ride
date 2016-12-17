import sys
import caffe
from copy import copy
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
	parser.add_argument('model',              type=str)
	parser.add_argument('--data',       '-d', type=str, default='data/BSDS300_test.mat')
	parser.add_argument('--img_num',    '-i', type=int, default=56)
	parser.add_argument('--holes',	    '-H', type=float, default=0.0)
	parser.add_argument('--momentum',	'-m', type=float, default=0.9)
	parser.add_argument('--lr',  		'-l', type=float, default=0.00001)
	parser.add_argument('--niter',  	'-N', type=int, default=10000)
	parser.add_argument('--path',       '-p', type=str, default='/home/cplab-ws1/ride/code/map_interpolate/')
	parser.add_argument('--mode',       '-q', type=str,   default='CPU', choices=['CPU', 'GPU'])
	parser.add_argument('--device',     '-D', type=int,   default=0)
	parser.add_argument('--margin',		'-M', type=int,	  default = 1)
	parser.add_argument('--size',		'-s', type=int, default= 256)
	parser.add_argument('--flip',	'-f', type=int,default=0)
	parser.add_argument('--ent_max', '-e', type=float, default=100.0)
	parser.add_argument('--resume','-r',type=int,default= -1)
	parser.add_argument('--patch_size', '-P', type=int,default=128)
	parser.add_argument('--index','-I',type=int,default=0)
	
	args = parser.parse_args(argv[1:])
	niter = args.niter
	lr = args.lr
	N = args.size

	# select CPU or GPU for caffe
	if args.mode.upper() == 'GPU':
		print "setting the GPU mode"
		caffe.set_mode_gpu()
		caffe.set_device(args.device)
	else:
		caffe.set_mode_cpu()

	path = args.path
	if args.index > -1:
		path += str(args.index)+'/'
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
		# images = loadmat(args.data)['data']
		# # load an image, 56 looks good
		# img = images[args.img_num,:args.size,:args.size]
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
	binary_mask = get_mask(img,args.margin, holes = args.holes)
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

	if args.resume == -1:
		init_img = np.copy(masked_images)
		for j in range(masked_images.shape[0]):
			num_dir  = 4
			est_img_arr = []
			sampled_img_arr =[]
			for i in range(num_dir): 
				print 'Sampling',j,i
				masked_img = masked_images[j,:,:,:].squeeze()[::(-1)**i,::(-1)**(i/2)]
				binary_mask = stacked_masks[j,:,:,:].squeeze()[::(-1)**i,::(-1)**(i/2)]
				est_img,sampled_img,ll = model.interpolate(masked_img,binary_mask,method = 'mode')
				est_img[est_img > 1.] = 1.
				est_img[est_img < 0.] = 0.

				est_img_arr.append(est_img[::(-1)**i,::(-1)**(i/2)])
				sampled_img_arr.append(sampled_img[::(-1)**i,::(-1)**(i/2)])
				mplimg.imsave(path+''+str(j)+'/sampled_img'+str(i), est_img[::(-1)**i,::(-1)**(i/2)], cmap = cm.gray,vmin=0,vmax=1)
			est_img_net = np.median(np.asarray(est_img_arr),axis =0)
			sampled_img_net = np.median(np.asarray(sampled_img_arr),axis =0)
			img_j = images[j,:,:,:].squeeze()
			ssim1 = ssim(est_img_net,img_j,dynamic_range=img_j.min()-img_j.max())
			psnr1 = psnr(est_img_net,img_j,dynamic_range=img_j.min()-img_j.max())
			print j,'Initial ssim',ssim1,'Initial psnr', psnr1
			init_img[j,:,:,:] = fd(est_img_net)
	
	elif args.resume == 0:
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
		f, grad_img, whitened_img = model.gradient_mul(init_img[:,::(-1)**j,::(-1)**(j/2),:],path=path,niter=i,ent_max = args.ent_max)

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

				mplimg.imsave(path+''+str(k)+'/cleaned_img'+str(i), cleaned_img, cmap = cm.gray,vmin=0,vmax=1)
				ssim1 = ssim(cleaned_img,img_k,dynamic_range=img_k.max()-img_k.min())
				psnr1 = psnr(cleaned_img,img_k,dynamic_range=img_k.max()-img_k.min())
				print k, 'ssim',ssim1,'psnr', psnr1 #, 'min',cleaned_img.min(),'max',cleaned_img.max()

		if (i%10==0):
			for k in range(images.shape[0]):
				fig2 = plt.figure(2)
				plt.imshow(grad_img[k,:,:,:].squeeze(),vmin = -0.02,vmax=0.02)
				plt.colorbar()
				plt.savefig(path+''+str(k)+'/grad_img'+str(i))
				plt.close(fig2)	

				fig3 = plt.figure(3)
				plt.imshow(whitened_img[k,:,:,:].squeeze(),vmin=-5.0,vmax=6.0)
				plt.colorbar()
				plt.savefig(path+''+str(k)+'/whitened_img'+str(i))
				plt.close(fig3)	

		if (i%50==0):
			np.save(path+'cleaned_img/'+str(i),init_img)

	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
