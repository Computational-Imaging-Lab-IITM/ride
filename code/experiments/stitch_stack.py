import numpy as np
from glob import glob
import matplotlib.image as mplimg
from skimage.measure import compare_ssim as ssim, compare_psnr as psnr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys
from skimage.color import rgb2gray
from scipy.io import loadmat

parser = ArgumentParser(sys.argv[0], description=__doc__)
parser.add_argument('--data',       '-d', type=str, default='data/BSDS_Cropped/img_data.mat')
parser.add_argument('--path',       '-p', type=str, default='/home/cplab-ws1/ride/code/map_interpolate/')
parser.add_argument('--index', '-I', type=int, default =-1)
parser.add_argument('--image','-i',type=int,default=0)
parser.add_argument('--size','-s',type=int,default=256)

args = parser.parse_args(sys.argv[1:])
path=args.path

if args.index>-1:
	path =path+str(args.index)+'/'
	print path
	#path=path+'Exp5/'

# load data
if args.data.lower()[-4:] in ['.gif', '.png', '.jpg', 'jpeg']:
	images = plt.imread(args.data)
	img = rgb2gray(images) 
	img = img[:args.size,:args.size]
	print 'img shape',img.shape
	#img = (img/255.0)  
    #vmin, vmax = 0, 255

else:
	images = loadmat(args.data)['data']
	# load an image, 56 looks good
	img = images[args.index,:args.size,:args.size].squeeze()
	print 'img shape', img.shape
	del images


mplimg.imsave(path+'original_img',img,cmap='gray',vmin=0,vmax=1)
init_img = np.load(path+'cleaned_img/'+str(args.image)+'.npy')
cleaned_img = np.zeros_like(img)
m =2
print init_img[:,m:-m,m:-m,:].shape
size = 130
count = np.zeros_like(img)
for i in range(init_img.shape[0]):
	m = 2
	print i
	cleaned_img[126*(i/2)+m:126*(i/2)+130-m,116*(i%2)+m:116*(i%2)+140-m] += init_img[i,m:-m,m:-m,:].squeeze()	
	count[126*(i/2)+m:126*(i/2)+130-m,116*(i%2)+m:116*(i%2)+140-m] +=1
count[count==0]=1
cleaned_img = cleaned_img/count
mplimg.imsave(path+'cleaned_img'+str(args.image),cleaned_img[m:-m,m:-m],cmap='gray',vmin=0,vmax=1)

ssim1 = ssim(cleaned_img[m:-m,m:-m],img[m:-m,m:-m],dynamic_range=img.min()-img.max())
psnr1 = psnr(cleaned_img[m:-m,m:-m],img[m:-m,m:-m],dynamic_range=img.min()-img.max())
print 'ssim',ssim1,'psnr',psnr1
