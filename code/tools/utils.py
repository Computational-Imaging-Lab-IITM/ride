# tools for me
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cmt.tools import imread, generate_data_from_image
import matplotlib.image as mplimg
from matplotlib import cm
from tools import mapp

import sys
import os
from numpy import zeros_like, asarray

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))
    
def fread(fpath):
	f = open(fpath, 'r')
	lines = f.readlines()
	lines = [line.strip().split(' ') for line in lines]
	flines = []
	for line in lines:
		l = [int(i) for i in line]
		flines.append(l)
	return flines
	
def td(a):
	return a.reshape(a.shape[1],a.shape[2])
		
def fd(a):
	return a.reshape(1,a.shape[0],a.shape[1],1)	
	
def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray

def _adjust_gradient(preconditioner, inputs, outputs):
	if preconditioner is None:
		raise RuntimeError('No preconditioner set.')

	#ip_shape = inputs.shape # (n, r, c, 12)
	#op_shape = outputs.shape

	#inputs = inputs.reshape(-1, inputs.shape[-1]).T # 12 x N
	#outputs = outputs.reshape(-1, outputs.shape[-1]).T # 1xN

	#inputs, outputs = self.preconditioner.adjust_gradient(inputs, outputs)

	pre_in = preconditioner.pre_in # 12 x 12
	#print (pre_in - pre_in.T).sum()
	pre_out = preconditioner.pre_out # 1
	predictor = preconditioner.predictor # 1x12

	inputs = np.dot(pre_in, inputs) # 12 x N
	tmp_var = np.dot(predictor, pre_in) # 1x12
	tmp_var = np.dot(pre_out,tmp_var)

	#print tmp_var.shape, outputs.shape
	tmp_var = np.dot(tmp_var.T, outputs) # 12xN
	#print tmp_var.shape

	inputs -= tmp_var # 12xN
	outputs = np.dot(pre_out,outputs) # 1xN

	#inputs = inputs.T.reshape(*ip_shape)
	#outputs = outputs.T.reshape(shape[0], shape[1], shape[2], -1)

	return inputs, outputs


def add_noise(img, noise_std):
	noisy_img = np.zeros(img.shape)
	noisy_img[:,:] = img[:,:]
	np.random.seed(0)
	noisy_img += noise_std*np.random.randn(img.shape[0],img.shape[1])

	return noisy_img


def _preprocess(images, input_mask, output_mask):
    """
    Extract causal neighborhoods from images.

    @type  images: C{ndarray}/C{list}
    @param images: array or list of images to process

    @rtype: C{tuple}
    @return: one array storing inputs (neighborhoods) and one array storing outputs (pixels)
    """

    def process(image):
        inputs, outputs = generate_data_from_image(
            image, input_mask, output_mask)
        inputs = asarray(
            inputs.T.reshape(
                image.shape[0] - input_mask.shape[0] + 1,
                image.shape[1] - input_mask.shape[1] + 1,
                -1), dtype='float32')
        outputs = asarray(
            outputs.T.reshape(
                image.shape[0] - input_mask.shape[0] + 1,
                image.shape[1] - input_mask.shape[1] + 1,
                -1), dtype='float32')
        return inputs, outputs

    inputs, outputs = zip(*mapp(process, images))

    return asarray(inputs), asarray(outputs)

def _precondition(preconditioner, inputs, outputs=None):
    """
    Remove any correlations within and between inputs and outputs (conditional whitening).

    @type  inputs: C{ndarray}
    @param inputs: pixel neighborhoods stored column-wise

    @type  outputs: C{ndarray}
    @param outputs: output pixels stored column-wise
    """

    shape = inputs.shape

    if outputs is None:
        if preconditioner is None:
            raise RuntimeError('No preconditioning possible.')

        inputs = inputs.reshape(-1, inputs.shape[-1]).T
        inputs = preconditioner(inputs)
        inputs = inputs.T.reshape(*shape)

        return inputs

    else:
        inputs = inputs.reshape(-1, inputs.shape[-1]).T
        outputs = outputs.reshape(-1, outputs.shape[-1]).T

        # avoids memory issues
        MAX_SAMPLES = 500000

        if preconditioner is None:
            if inputs.shape[1] > MAX_SAMPLES:
                idx = random_select(MAX_SAMPLES, inputs.shape[1])
                preconditioner = WhiteningPreconditioner(inputs[:, idx], outputs[:, idx])
            else:
                preconditioner = WhiteningPreconditioner(inputs, outputs)

        for b in range(0, inputs.shape[1], MAX_SAMPLES):
            inputs[:, b:b + MAX_SAMPLES], outputs[:, b:b + MAX_SAMPLES] = \
                preconditioner(inputs[:, b:b + MAX_SAMPLES], outputs[:, b:b + MAX_SAMPLES])


        inputs = inputs.T.reshape(*shape)
        outputs = outputs.T.reshape(shape[0], shape[1], shape[2], -1)

        return inputs, outputs        



def tv_norm(x, beta=2.0, verbose=False, operator='naive'):
	"""
	Compute the total variation norm and its gradient.

	The total variation norm is the sum of the image gradient
	raised to the power of beta, summed over the image.
	We approximate the image gradient using finite differences.
	We use the total variation norm as a regularizer to encourage
	smoother images.
	Inputs:
	- x: numpy array of shape (1, C, H, W)
	Returns a tuple of:
	- loss: Scalar giving the value of the norm
	- dx: numpy array of shape (1, C, H, W) giving gradient of the loss
		with respect to the input x.
	"""
	assert x.shape[0] == 1
	if operator == 'naive':
		x_diff = x[:, :, :-1, :-1] - x[:, :, :-1, 1:]
		y_diff = x[:, :, :-1, :-1] - x[:, :, 1:, :-1]
	elif operator == 'sobel':
		x_diff  =  x[:, :, :-2, 2:]  + 2 * x[:, :, 1:-1, 2:]  + x[:, :, 2:, 2:]
		x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
		y_diff  =  x[:, :, 2:, :-2]  + 2 * x[:, :, 2:, 1:-1]  + x[:, :, 2:, 2:]
		y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
	elif operator == 'sobel_squish':
		x_diff  =  x[:, :, :-2, 1:-1]  + 2 * x[:, :, 1:-1, 1:-1]  + x[:, :, 2:, 1:-1]
		x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
		y_diff  =  x[:, :, 1:-1, :-2]  + 2 * x[:, :, 1:-1, 1:-1]  + x[:, :, 1:-1, 2:]
		y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
	else:
		assert False, 'Unrecognized operator %s' % operator
		
	grad_norm2 = x_diff ** 2.0 + y_diff ** 2.0
	grad_norm2[grad_norm2 < 1e-3] = 1e-3
	grad_norm_beta = grad_norm2 ** (beta / 2.0)
	loss = np.sum(grad_norm_beta)
	dgrad_norm2 = (beta / 2.0) * grad_norm2 ** (beta / 2.0 - 1.0)
	dx_diff = 2.0 * x_diff * dgrad_norm2
	dy_diff = 2.0 * y_diff * dgrad_norm2
	dx = np.zeros_like(x)
	
	if operator == 'naive':
		dx[:, :, :-1, :-1] += dx_diff + dy_diff
		dx[:, :, :-1, 1:] -= dx_diff
		dx[:, :, 1:, :-1] -= dy_diff
	elif operator == 'sobel':
		dx[:, :, :-2, :-2] += -dx_diff - dy_diff
		dx[:, :, :-2, 1:-1] += -2 * dy_diff
		dx[:, :, :-2, 2:] += dx_diff - dy_diff
		dx[:, :, 1:-1, :-2] += -2 * dx_diff
		dx[:, :, 1:-1, 2:] += 2 * dx_diff
		dx[:, :, 2:, :-2] += dy_diff - dx_diff
		dx[:, :, 2:, 1:-1] += 2 * dy_diff
		dx[:, :, 2:, 2:] += dx_diff + dy_diff
	elif operator == 'sobel_squish':
		dx[:, :, :-2, :-2] += -dx_diff - dy_diff
		dx[:, :, :-2, 1:-1] += dx_diff -2 * dy_diff
		dx[:, :, :-2, 2:] += -dy_diff
		dx[:, :, 1:-1, :-2] += -2 * dx_diff + dy_diff
		dx[:, :, 1:-1, 1:-1] += 2 * dx_diff + 2 * dy_diff
		dx[:, :, 1:-1, 2:] += dy_diff
		dx[:, :, 2:, :-2] += -dx_diff
		dx[:, :, 2:, 1:-1] += dx_diff

  
	def helper(name, x):
		num_nan = np.isnan(x).sum()
		num_inf = np.isinf(x).sum()
		num_zero = (x == 0).sum()
		print '%s: NaNs: %d infs: %d zeros: %d' % (name, num_nan, num_inf, num_zero)
  
	if verbose:
		print '-' * 40
		print 'tv_norm debug output'
		helper('x', x)
		helper('x_diff', x_diff)
		helper('y_diff', y_diff)
		helper('grad_norm2', grad_norm2)
		helper('grad_norm_beta', grad_norm_beta)
		helper('dgrad_norm2', dgrad_norm2)
		helper('dx_diff', dx_diff)
		helper('dy_diff', dy_diff)
		helper('dx', dx)
		print
  
	return loss, dx


def group_denoise(img,n,l,N,s,b,pid,p,model,input_mask,output_mask,preconditioner):
	'''
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--noise_std',  '-n', type=float, default=0.1)
	parser.add_argument('--lr',  		'-l', type=float, default=0.01)
	parser.add_argument('--niter',  	'-N', type=int, default=100)
	parser.add_argument('--patch_size', '-s', type=int, default=9)
	parser.add_argument('--nbd_size',  	'-b', type=int, default=78)
	parser.add_argument('--pid',  	    '-I', type=int, default=None)
	parser.add_argument('--path',       '-p', type=str, default='ada_mcgsm/denoise/')

	args = parser.parse_args(argv[1:])
	'''
	niter = N
	lr = l
	if not os.path.exists(p):
		os.makedirs(p)
	path = p	
	denoise_path = p + 'denoise_' + str(pid)+'/'
	if not os.path.exists(denoise_path):
		os.makedirs(denoise_path)

	num_samples = 1
	nbd_list = [3]
	llw = 1.0

	lines = fread('clusters3.txt')
	
	mplimg.imsave(path+'given_img', img, vmin=0, vmax=1, cmap = cm.gray)	
	print 'img shape', img.shape, img.max(), img.min()
	
	images = img.reshape([1,img.shape[0],img.shape[1],1])
	
	# noise stat
	noise = {}
	noise['std'] = n
	noise_std = n
	noise['mean'] = 0
	noise['fill'] = None#fill_shape
	
	# adding noise	
	noisy_img = add_noise(img, noise, input_mask)
	print 'max noisy', noisy_img.max()
	mplimg.imsave(path+'noisy_img', noisy_img, cmap = cm.gray)	
	print noisy_img.shape
		
	init_img = np.copy(noisy_img)
	
	h = img.shape[0]
	w = img.shape[1]
	N = b
	patch_size = s
	width = w + 2*N
	
	tmp_img = np.zeros(img.shape)
	tmp_noisy_img = np.zeros(img.shape)


	inputs = np.zeros((sum(sum(input_mask)), 1))
	outputs = np.zeros((1, 1))
	group_loc = []	
	group = []

	if(len(lines[pid])>0):	
		_group = np.array(lines[pid]).astype('int')
		
		ac = 0
		for g in _group:
			group.append(g)
			r = g/width - N 
			c = g%width - N
			if(r>0 and r+patch_size<=h and c>0 and c+patch_size<=w):
				tmp_img[r:r+patch_size,c:c+patch_size] = img[r:r+patch_size,c:c+patch_size]
				tmp_noisy_img[r:r+patch_size,c:c+patch_size] = noisy_img[r:r+patch_size,c:c+patch_size]
			
				tmp = noisy_img[r:r+patch_size,c:c+patch_size]
				for i in nbd_list:#np.random.choice(range(1+patch_size/2),num_samples):
					#print i,
					t = tmp[i:i+1+patch_size/2,:]
					inputs = np.concatenate((inputs, t[input_mask].reshape(-1,1)),axis=1)
					outputs = np.concatenate((outputs, t[output_mask].reshape(-1,1)),axis=1)
					group_loc.append([r+i,c])
				ac+=1 			
				
	inputs = np.delete(inputs, range(1), 1)
	outputs = np.delete(outputs, range(1), 1)				
			
	group = np.array(group).astype(int)

	mplimg.imsave(path+'img_'+str(pid), tmp_img, vmin=0, vmax=1, cmap = cm.gray)	
	mplimg.imsave(path+'noisy_img_'+str(pid), tmp_noisy_img, vmin=0, vmax=1, cmap = cm.gray)	

	print 'num elements', inputs.shape[1], len(group_loc)
	
	init_img = np.copy(tmp_noisy_img)
		

	prev_grad = zeros_like(init_img)
	normalizer = inputs.shape[1]#/3
	for k in range(niter):
		print 'iter', k, 'loglik', 
			
		# get inputs and outputs
		inputs, outputs = preconditioner(inputs, outputs)
		#print inputs.shape, outputs.shape
		
		# compute gradients
		df_dh, df_dy, loglik = model._data_gradient(inputs, outputs)
		#print df_dh.shape, df_dy.shape
		
		df_dh = df_dh / normalizer
		df_dy = df_dy / normalizer
		
		# average log-likelihood
		f = sum(loglik) / inputs.shape[1] 
		print f,
		
		df_dh, df_dy = _adjust_gradient(preconditioner, df_dh, df_dy)	
		
		gradient = zeros_like(init_img)
		ll_img = zeros_like(init_img)
		
		for idx, g in enumerate(group_loc):
			r = g[0]	
			c = g[1]
			cid = 0
			#print r,c, idx
			for _r in range(patch_size/2 + 1):
				for _c in range(patch_size):			
					gradient[r+_r,c+_c] += df_dh[cid,idx]
					cid+=1
					if cid>sum(sum(input_mask))-1:	
						break
			
			gradient[r+patch_size/2,c+patch_size/2] += df_dy[0,idx]
			ll_img[r+patch_size/2,c+patch_size/2] += loglik[0,idx]

		likelihood = tmp_noisy_img - init_img
		likelihood = llw*likelihood / (noise_std**2) / normalizer
		
		print 'grad', np.abs(gradient).sum(), 'll_grad', np.abs(likelihood).sum(), 
				
		
		prev_grad = 0.9*prev_grad + gradient + likelihood
		
		init_img += lr*prev_grad			
		'''
		if(k%5==0 and k!=0):
			plt.figure(1)
			plt.imshow(gradient,vmin=-0.02,vmax=0.02)
			plt.colorbar()
			plt.savefig(path+'grad_img_'+str(k))
			plt.close()

			plt.figure(2)
			plt.imshow(ll_img,vmin=-1.0,vmax=1.0)
			plt.colorbar()
			plt.savefig(path+'ll_img_'+str(k))
			plt.close()
		'''
		#init_img = init_img.clip(0,1)
		
		diff = tmp_img - init_img.reshape(*img.shape)
		mse = np.mean(np.square(diff))
		psnr = -10*np.log10(mse)
		print 'psnr', psnr
		fid = open(path+'mcgsm_psnr.log','a')
		fid.write(str(f)+' '+str(psnr)+'\n')	
		fid.close()
						
		if(k%20==0):
			#print 'saving'		
			cleaned_img = np.copy(init_img)
			mplimg.imsave(denoise_path+'t_img'+str(k), cleaned_img, cmap = cm.gray)
			
		#if(k%10==0 and k!=0 and llw>1):
		#	llw = 0.9*llw
				
		#if(k%100==0 and k!=0):
		#	lr = 0.95*lr

		inputs = zeros_like(inputs)
		outputs = zeros_like(outputs)

		ac = 0
		group_loc = []
		for g in group:
			r = g/width - N 
			c = g%width - N
		
			if(r>0 and r+patch_size<=h and c>0 and c+patch_size<=w):
				tmp = np.copy(init_img[r:r+patch_size,c:c+patch_size])
				for i in np.random.choice(range(1+patch_size/2),num_samples):
				#for i in nbd_list:
					t = tmp[i:i+1+patch_size/2,:]
					inputs[:,ac] = t[input_mask].reshape(sum(sum(input_mask)))
					outputs[:,ac] = t[output_mask].reshape(1)
					group_loc.append([r+i,c])
				ac+=1 		
						
	cleaned_img = np.copy(init_img)		
	
	diff = tmp_img - cleaned_img
	mse = np.mean(np.square(diff))
	print 'mse', mse
	psnr = -10*np.log10(mse)
	print psnr

	diff = tmp_img - tmp_noisy_img
	mse = np.mean(np.square(diff))
	print 'mse', mse
	psnr = -10*np.log10(mse)
	print 'noise', psnr		

	mplimg.imsave(path+'cleaned_img_'+str(pid), cleaned_img, cmap = cm.gray)

	return cleaned_img


