__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

import os
#import h5py

from collections import defaultdict
from cmt.models import MCGSM
from cmt.tools import generate_masks, generate_data_from_image, sample_image
from cmt.utils import random_select
from cmt.transforms import WhiteningPreconditioner
from numpy import asarray, log, sum, zeros_like, mean, any, ceil, min, max, isnan, zeros,tanh
from numpy import square, sqrt, power
from numpy.linalg import norm
from mapp import mapp
from time import time
from sfo import SFO
from .slstm import SLSTM
import os.path
from scipy.io import savemat
import numpy as np
from copy import copy,deepcopy
import scipy.stats
from sklearn import mixture
import matplotlib.image as mplimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from matplotlib import cm

class RIDE(object):
    """
    An implementation of the recurrent image density estimator (RIDE).

    B{References:}
        - Theis, L. and Bethge, M. (2015). I{Generative Image Modeling Using Spatial LSTMs.}
    """

    # maximum batch size used by Caffe internally
    MAX_BATCH_SIZE = 200

    def __init__(self,
        num_channels=1,
        num_hiddens=10,
        num_components=8,
        num_scales=4,
        num_features=16,
        num_layers=1,
        nb_size=5,
        nonlinearity='TanH',
        verbosity=1,
        extended=False,
        input_mask=None,
        output_mask=None):
        """
        @type  num_channels: C{int}
        @param num_channels: dimensionality of each pixel

        @type  num_hiddens: C{int}
        @param num_hiddens: number of LSTM units in each spatial LSTM layer

        @type  num_components: C{int}
        @param num_components: number of mixture components used by the MCGSM

        @type  num_scales: C{int}
        @param num_scales: number of scales used by the MCGSM

        @type  num_features: C{int}
        @param num_features: number of quadratic features used by the MCGSM

        @type  num_layers: C{int}
        @param num_layers: number of layers of spatial LSTM units

        @type  nb_size: C{int}
        @param nb_size: controls the neighborhood of pixels read from an image

        @type  nonlinearity: C{str}
        @param nonlinearity: nonlinearity used by spatial LSTM (e.g., TanH, ReLU)

        @type  verbosity: C{int}
        @param verbosity: controls how much information is printed during training, etc.

        @type  extended: C{bool}
        @param extended: use previous memory states as additional inputs to LSTM (more parameters)

        @type  input_mask C{ndarray}
        @param input_mask: Boolean mask used to define custom input neighborhood of pixels

        @type  output_mask C{ndarray}
        @param output_mask: determines the position of the output pixel relative to the neighborhood
        """

        self.verbosity = verbosity

        self.num_channels = num_channels
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.extended = extended

        self.input_mask, self.output_mask = generate_masks([nb_size] * num_channels)

        if input_mask is not None:
            self.input_mask = input_mask
            print 'input_mask',input_mask.shape
            print 'input_channels',sum(input_mask)
        if output_mask is not None:
            self.output_mask = output_mask
            print 'output_mask',output_mask.shape
            self.num_channels = sum(self.output_mask)
            print 'num_channels',num_channels

        self.slstm = [None] * num_layers
        #print sum(input_mask)
        #print 'Creating LSTM'
        for l in range(self.num_layers):
            self.slstm[l] = SLSTM(   #Manual change
                    num_rows=11,
                    num_cols=11,
                    num_channels=sum(self.input_mask),
                    num_hiddens=self.num_hiddens,
                    batch_size=64,
                    nonlinearity=self.nonlinearity,
                    extended=self.extended,
                    slstm=None,
                    verbosity=self.verbosity)
        #print 'Creating MCGSM'
        self.mcgsm = MCGSM(
            dim_in=self.num_hiddens,
            dim_out=self.num_channels,
            num_components=num_components,
            num_scales=num_scales,
            num_features=num_features)
        self.preconditioner = None



    def add_layer(self):
        """
        Add another spatial LSTM to the network and reinitialize MCGSM.
        """

        self.num_layers += 1

        # reinitialize MCGSM
        self.mcgsm = MCGSM(
            dim_in=self.num_hiddens,
            dim_out=self.num_channels,
            num_components=self.mcgsm.num_components,
            num_scales=self.mcgsm.num_scales,
            num_features=self.mcgsm.num_features)

        # add slot for another layer
        self.slstm.append(None)



    def _precondition(self, inputs, outputs=None):
        """
        Remove any correlations within and between inputs and outputs (conditional whitening).

        @type  inputs: C{ndarray}
        @param inputs: pixel neighborhoods stored column-wise

        @type  outputs: C{ndarray}
        @param outputs: output pixels stored column-wise
        """

        shape = inputs.shape

        if outputs is None:
            if self.preconditioner is None:
                raise RuntimeError('No preconditioning possible.')

            inputs = inputs.reshape(-1, inputs.shape[-1]).T
            inputs = self.preconditioner(inputs)
            inputs = inputs.T.reshape(*shape)

            return inputs

        else:
            inputs = inputs.reshape(-1, inputs.shape[-1]).T
            outputs = outputs.reshape(-1, outputs.shape[-1]).T

            # avoids memory issues
            MAX_SAMPLES = 500000

            if self.preconditioner is None:
                if inputs.shape[1] > MAX_SAMPLES:
                    idx = random_select(MAX_SAMPLES, inputs.shape[1])
                    self.preconditioner = WhiteningPreconditioner(inputs[:, idx], outputs[:, idx])
                else:
                    self.preconditioner = WhiteningPreconditioner(inputs, outputs)

            for b in range(0, inputs.shape[1], MAX_SAMPLES):
                inputs[:, b:b + MAX_SAMPLES], outputs[:, b:b + MAX_SAMPLES] = \
                    self.preconditioner(inputs[:, b:b + MAX_SAMPLES], outputs[:, b:b + MAX_SAMPLES])


            inputs = inputs.T.reshape(*shape)
            outputs = outputs.T.reshape(shape[0], shape[1], shape[2], -1)

            return inputs, outputs



    def _precondition_inverse(self, inputs, outputs=None):
        """
        Reintroduce correlations removed by conditional whitening.

        @type  inputs: C{ndarray}
        @param inputs: pixel neighborhoods stored column-wise

        @type  outputs: C{ndarray}
        @param outputs: output pixels stored column-wise
        """

        if self.preconditioner is None:
            raise RuntimeError('No preconditioner set.')

        shape = inputs.shape

        if outputs is None:
            inputs = inputs.reshape(-1, inputs.shape[-1]).T
            inputs = self.preconditioner.inverse(inputs)
            inputs = inputs.T.reshape(*shape)

            return inputs

        else:
            inputs = inputs.reshape(-1, inputs.shape[-1]).T
            outputs = outputs.reshape(-1, outputs.shape[-1]).T

            inputs, outputs = self.preconditioner.inverse(inputs, outputs)

            inputs = inputs.T.reshape(*shape)
            outputs = outputs.T.reshape(shape[0], shape[1], shape[2], -1)

            return inputs, outputs



    def _adjust_gradient(self, inputs, outputs):
        """
        Adjust gradients to take into account preconditioning.

        @type  inputs: C{ndarray}
        @param inputs: gradient with respect to conditionally whitened inputs

        @type  outputs: C{ndarray}
        @param outputs: gradient with respect to conditionally whitened outputs
        """

        if self.preconditioner is None:
            raise RuntimeError('No preconditioner set.')

        shape = inputs.shape # (n, r, c, 12)

        inputs = inputs.reshape(-1, inputs.shape[-1]).T # 12 x N
        outputs = outputs.reshape(-1, outputs.shape[-1]).T # 1xN

        #inputs, outputs = self.preconditioner.adjust_gradient(inputs, outputs)

        pre_in = self.preconditioner.pre_in # 12 x 12
        pre_out = self.preconditioner.pre_out # 1 x 1
        predictor = self.preconditioner.predictor # 1x12

        inputs = np.dot(pre_in, inputs) # 12 x N
        tmp_var = np.dot(pre_out,(np.dot(predictor, pre_in))) # 1x12
        tmp_var = np.dot(tmp_var.T, outputs) # 12xN
        #print tmp_var.shape

        inputs -= tmp_var # 12xN
        outputs = np.dot(pre_out,outputs) # 1xN


        inputs = inputs.T.reshape(*shape)
        outputs = outputs.T.reshape(shape[0], shape[1], shape[2], -1)
        
        return inputs, outputs



    def _preprocess(self, images):
        """
        Extract causal neighborhoods from images.

        @type  images: C{ndarray}/C{list}
        @param images: array or list of images to process

        @rtype: C{tuple}
        @return: one array storing inputs (neighborhoods) and one array storing outputs (pixels)
        """

        def process(image):
            inputs, outputs = generate_data_from_image(
                image, self.input_mask, self.output_mask)
            inputs = asarray(
                inputs.T.reshape(
                    image.shape[0] - self.input_mask.shape[0] + 1,
                    image.shape[1] - self.input_mask.shape[1] + 1,
                    -1), dtype='float32')
            outputs = asarray(
                outputs.T.reshape(
                    image.shape[0] - self.input_mask.shape[0] + 1,
                    image.shape[1] - self.input_mask.shape[1] + 1,
                    -1), dtype='float32')
            return inputs, outputs

        inputs, outputs = zip(*mapp(process, images))

        return asarray(inputs), asarray(outputs)



    def loglikelihood(self, images):
        """
        Returns a log-likelihood for each reachable pixel (in nats).

        @type  images: C{ndarray}/C{list}
        @param images: array or list of images for which to evaluate log-likelihood

        @rtype: C{ndarray}
        @return: an array of log-likelihoods for each image and predicted pixel
        """
        #print 'Shape of images in loglik',images.shape
        inputs, outputs = self._preprocess(images)

        if self.preconditioner is not None:
            if self.verbosity > 0:
                print 'Computing Jacobian...'

            logjacobian = self.preconditioner.logjacobian(
                inputs.reshape(-1, sum(self.input_mask)).T,
                outputs.reshape(-1, self.num_channels).T)

            if self.verbosity > 0:
                print 'Preconditioning...'

            # remove correlations
            inputs, outputs = self._precondition(inputs, outputs)
            #print 'Done Preconditioning'

        else:
            logjacobian = 0.

        # compute hidden unit activations
        hiddens = inputs
        #print hiddens.shape, outputs.shape

        #batch_size = min([hiddens.shape[0], self.MAX_BATCH_SIZE])
        batch_size = min([hiddens.shape[0], 32])
        #print 'batch_size',batch_size

        if self.verbosity > 0:
            print 'Computing hidden states...'
            # print 'check ----'

        for l in range(self.num_layers):
            # create SLSTM
            #print 'creating lstm layer', l
            #print 'hidden shape',hiddens.shape
            if self.slstm[l].num_rows != hiddens.shape[1] \
                or self.slstm[l].num_cols != hiddens.shape[2] \
                or self.slstm[l].batch_size != batch_size:
                #print 'creating now'
                self.slstm[l] = SLSTM(
                    num_rows=hiddens.shape[1],
                    num_cols=hiddens.shape[2],
                    num_channels=hiddens.shape[3],
                    num_hiddens=self.num_hiddens,
                    batch_size=batch_size,
                    nonlinearity=self.nonlinearity,
                    extended=self.extended,
                    slstm=self.slstm[l],
                    verbosity=self.verbosity)
            #print 'forwarding the data'
            hiddens = self.slstm[l].forward(hiddens)
            #print 'Size of hiddens given to mcgsm',hiddens.shape

        if self.verbosity > 0:
            print 'Computing likelihood...'

        # evaluate log-likelihood
        loglik = self.mcgsm.loglikelihood(
            hiddens.reshape(-1, self.num_hiddens).T,
            outputs.reshape(-1, self.num_channels).T) + logjacobian
        #print loglik
        return loglik.reshape(hiddens.shape[0], hiddens.shape[1], hiddens.shape[2])



    def evaluate(self, images):
        """
        Computes the average negative log-likelihood in bits per pixel.

        @type  images: C{ndarray}/C{list}
        @param images: an array or list of test images

        @rtype: C{float}
        @return: average negative log-likelihood in bits per pixel
        """

        return -mean(self.loglikelihood(images)) / log(2.) / self.num_channels



    def train(self, images,
            batch_size=50,
            num_epochs=20,
            method='SGD',
            train_means=False,
            train_top_layer=False,
            momentum=0.9,
            learning_rate=1.,
            decay1=0.9,
            decay2=0.999,
            precondition=True,
            save_grad=None,
            grad_fname=None):
        """
        Train model via stochastic gradient descent (SGD) or sum-of-functions optimizer (SFO).

        @type  images: C{ndarray}/C{list}
        @param images: an array or a list of training images (e.g., Nx32x32x3)

        @type  batch_size: C{int}
        @param batch_size: batch size used by SGD

        @type  num_epochs: C{int}
        @param num_epochs: number of passes through the training set

        @type  method: C{str}
        @param method: either 'SGD', 'SFO', or 'ADAM'

        @type  train_means: C{bool}
        @param train_means: whether or not to optimize the mean parameters of the MCGSM

        @type  train_top_layer: C{bool}
        @param train_top_layer: if true, only the MCGSM and spatial LSTM at the top layer is trained

        @type  momentum: C{float}
        @param momentum: momentum rate used by SGD

        @type  learning_rate: C{float}
        @param learning_rate: learning rate used by SGD

        @type  decay1: C{float}
        @param decay1: hyperparameter used by ADAM

        @type  decay2: C{float}
        @param decay2: hyperparameter used by ADAM

        @type  precondition: C{bool}
        @param precondition: whether or not to perform conditional whitening

        @rtype: C{list}
        @return: evolution of negative log-likelihood (bits per pixel) over the training
        """

        if save_grad is not None:
            if grad_fname is None:
                raise ValueError('Please specify the filename to save gradients.')

        if images.shape[1] < self.input_mask.shape[0] or images.shape[2] < self.input_mask.shape[1]:
            raise ValueError('Images too small.')

        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)
        print 'train stat: input shape', inputs.shape, 'output shape:', outputs.shape
        if precondition:
            if self.verbosity > 0:
                print 'Preconditioning...'

            # remove correlations
            inputs, outputs = self._precondition(inputs, outputs)

        # indicates which layers will be trained
        train_layers = [self.num_layers - 1] if train_top_layer else range(self.num_layers)

        if self.verbosity > 0:
            print 'Creating SLSTMs...'

        # create SLSTMs
        for l in range(self.num_layers):
            self.slstm[l] = SLSTM(
                num_rows=inputs.shape[1],
                num_cols=inputs.shape[2],
                num_channels=inputs.shape[3] if l < 1 else self.num_hiddens,
                num_hiddens=self.num_hiddens,
                batch_size=min([batch_size, self.MAX_BATCH_SIZE]),
                nonlinearity=self.nonlinearity,
                extended=self.extended,
                slstm=self.slstm[l],
                verbosity=self.verbosity)
            print 'done creating SLSTM'
        # compute loss function and its gradient
        def f_df(params, idx):
            # set model parameters
            for l in train_layers:
                self.slstm[l].set_parameters(params['slstm'][l])
            self.mcgsm._set_parameters(params['mcgsm'], {'train_means': train_means})

            # select batch and compute hidden activations
            Y = outputs[idx:idx + batch_size]
            H = inputs[idx:idx + batch_size]

            for l in range(self.num_layers):
                H = self.slstm[l].forward(H)

            # form inputs to MCGSM
            H_flat = H.reshape(-1, self.num_hiddens).T
            Y_flat = Y.reshape(-1, self.num_channels).T

            norm_const = -H_flat.shape[1]

            # compute gradients
            df_dh, _, loglik = self.mcgsm._data_gradient(H_flat, Y_flat)
            df_dh = df_dh.T.reshape(*H.shape) / norm_const

            # average log-likelihood
            # print 'loglik', loglik
            # print sum(loglik) / H_flat.shape[1]
            f = sum(loglik) / norm_const
            
            df_dtheta = {}
            df_dtheta['slstm'] = [0.] * self.num_layers

            for l in range(self.num_layers)[::-1]:
                if l not in train_layers:
                    break
                if l > min(train_layers):
                    # derivative with respect to inputs of layer l are derivatives
                    # of hidden states of layer l - 1
                    df_dtheta['slstm'][l] = self.slstm[l].backward(df_dh, force_backward=True)
                    df_dh = df_dtheta['slstm'][l]['inputs']
                    del df_dtheta['slstm'][l]['inputs']

                else:
                    # no need to compute derivatives with respect to input units
                    df_dtheta['slstm'][l] = self.slstm[l].backward(df_dh)

            # compute gradient of MCGSM
            df_dtheta['mcgsm'] = self.mcgsm._parameter_gradient(H_flat, Y_flat,
                parameters={'train_means': train_means}) * log(2.) * self.mcgsm.dim_out

            return f, df_dtheta

        # collect current parameters
        params = {}
        params['slstm'] = [0.] * self.num_layers
        for l in range(self.num_layers)[::-1]:
            if l not in train_layers:
                break
            params['slstm'][l] = self.slstm[l].parameters()
        params['mcgsm'] = self.mcgsm._parameters({'train_means': train_means})

        # a start index for each batch
        start_indices = range(
            0, inputs.shape[0] - batch_size + 1, batch_size)

        if self.verbosity > 0:
            print 'Training...'

        if method.upper() == 'SFO':
            try:
                # optimize using sum-of-functions optimizer
                optimizer = SFO(f_df, params, start_indices, display=self.verbosity)
                params_opt = optimizer.optimize(num_passes=num_epochs)

                # set model parameters
                for l in range(self.num_layers):
                    self.slstm[l].set_parameters(params_opt['slstm'][l])
                self.mcgsm._set_parameters(params_opt['mcgsm'], {'train_means': train_means})

            except KeyboardInterrupt:
                pass

            return optimizer.hist_f_flat

        elif method.upper() == 'SGD':
            loss = []
            diff = {
                'slstm': [0.] * self.num_layers,
                'mcgsm': zeros_like(params['mcgsm'])}

            for l in train_layers:
                diff['slstm'][l] = {}
                for key in params['slstm'][l]:
                    if save_grad is not None:
                        #print 'creating grad txts'
                        fname = grad_fname+'slstm'+str(l)+'_'+key
                        pname = grad_fname+'slstm'+str(l)+'_para_'+key
                        #if(not os.path.isfile(fname+'.txt')):
                        g = open(fname+'.txt','w')
                        g.close()
                        p = open(pname+'.txt','w')
                        p.close()
                    diff['slstm'][l][key] = zeros_like(params['slstm'][l][key])

            for n in range(num_epochs):
                for b in range(0, inputs.shape[0] - batch_size + 1, batch_size):
                    # compute gradients
                    f, df = f_df(params, b)
                    
                    loss.append(f / log(2.) / self.num_channels)


                    # update SLSTM parameters
                    for l in train_layers:
                        for key in params['slstm'][l]:
                            diff['slstm'][l][key] = momentum * diff['slstm'][l][key] - df['slstm'][l][key]
                            #print type(diff['slstm'][l][key]),diff['slstm'][l][key].shape
                            if save_grad is not None:
                                fname = grad_fname+'slstm'+str(l)+'_'+key
                                #print fname
                                f = open(fname+'.txt','a')
                                l2_norm = norm(diff['slstm'][l][key])
                                f.write(str(l2_norm)+'\n')
                                f.close()
                            params['slstm'][l][key] = params['slstm'][l][key] + learning_rate * diff['slstm'][l][key]
                            #save param norms
                            pname = grad_fname+'slstm'+str(l)+'_para_'+key
                            p = open(pname+'.txt','a')
                            l2_norm = norm(params['slstm'][l][key])
                            p.write(str(l2_norm)+'\n')
                            p.close()

                    # update MCGSM parameters
                    diff['mcgsm'] = momentum * diff['mcgsm'] - df['mcgsm']
                    params['mcgsm'] = params['mcgsm'] + learning_rate * diff['mcgsm']

                    if self.verbosity > 0:
                        print '{0:>5} {1:>10.4f} {2:>10.4f}'.format(
                            n, loss[-1], mean(loss[-max([10, 20000 // batch_size]):]))
            return loss

        elif method.upper() == 'ADAM':
            loss = []
            diff_mean = {
                'slstm': [0.] * self.num_layers,
                'mcgsm': zeros_like(params['mcgsm'])}
            diff_sqrd = {
                'slstm': [0.] * self.num_layers,
                'mcgsm': zeros_like(params['mcgsm'])}

            for l in train_layers:
                diff_mean['slstm'][l] = {}
                diff_sqrd['slstm'][l] = {}
                for key in params['slstm'][l]:
                    diff_mean['slstm'][l][key] = zeros_like(params['slstm'][l][key])
                    diff_sqrd['slstm'][l][key] = zeros_like(params['slstm'][l][key])

            # step counter
            t = 1

            for n in range(num_epochs):
                for b in range(0, inputs.shape[0] - batch_size + 1, batch_size):
                    # compute gradients
                    f, df = f_df(params, b)

                    loss.append(f / log(2.) / self.num_channels)

                    # include bias correction in step width
                    step_width = learning_rate / (1. - power(decay1, t)) * sqrt(1. - power(decay2, t))
                    t += 1

                    # update SLSTM parameters
                    for l in train_layers:
                        for key in params['slstm'][l]:
                            diff_mean['slstm'][l][key] = decay1 * diff_mean['slstm'][l][key] \
                                + (1. - decay1) * df['slstm'][l][key]
                            diff_sqrd['slstm'][l][key] = decay2 * diff_sqrd['slstm'][l][key] \
                                + (1. - decay2) * square(df['slstm'][l][key])

                            params['slstm'][l][key] = params['slstm'][l][key] - \
                                step_width * diff_mean['slstm'][l][key] / (1e-8 + sqrt(diff_sqrd['slstm'][l][key]))

                    # update MCGSM parameters
                    diff_mean['mcgsm'] = decay1 * diff_mean['mcgsm'] + (1. - decay1) * df['mcgsm']
                    diff_sqrd['mcgsm'] = decay2 * diff_sqrd['mcgsm'] + (1. - decay2) * square(df['mcgsm'])
                    params['mcgsm'] = params['mcgsm'] - \
                        step_width * diff_mean['mcgsm'] / (1e-8 + sqrt(diff_sqrd['mcgsm']))

                    if self.verbosity > 0:
                        print '{0:>5} {1:>10.4f} {2:>10.4f}'.format(
                            n,
                            loss[-1],
                            mean(loss[-max([10, 20000 // batch_size]):]))

            return loss

        else:
            raise ValueError('Unknown method \'{0}\'.'.format(method))

    def train_noisy(self, images,noisy_images,
            batch_size=50,
            num_epochs=20,
            method='SGD',
            train_means=False,
            train_top_layer=False,
            momentum=0.9,
            learning_rate=1.,
            decay1=0.9,
            decay2=0.999,
            precondition=True,
            save_grad=None,
            grad_fname=None):
        """
        Train model via stochastic gradient descent (SGD) or sum-of-functions optimizer (SFO).

        @type  images: C{ndarray}/C{list}
        @param images: an array or a list of training images (e.g., Nx32x32x3)

        @type  batch_size: C{int}
        @param batch_size: batch size used by SGD

        @type  num_epochs: C{int}
        @param num_epochs: number of passes through the training set

        @type  method: C{str}
        @param method: either 'SGD', 'SFO', or 'ADAM'

        @type  train_means: C{bool}
        @param train_means: whether or not to optimize the mean parameters of the MCGSM

        @type  train_top_layer: C{bool}
        @param train_top_layer: if true, only the MCGSM and spatial LSTM at the top layer is trained

        @type  momentum: C{float}
        @param momentum: momentum rate used by SGD

        @type  learning_rate: C{float}
        @param learning_rate: learning rate used by SGD

        @type  decay1: C{float}
        @param decay1: hyperparameter used by ADAM

        @type  decay2: C{float}
        @param decay2: hyperparameter used by ADAM

        @type  precondition: C{bool}
        @param precondition: whether or not to perform conditional whitening

        @rtype: C{list}
        @return: evolution of negative log-likelihood (bits per pixel) over the training
        """

        if save_grad is not None:
            if grad_fname is None:
                raise ValueError('Please specify the filename to save gradients.')

        if images.shape[1] < self.input_mask.shape[0] or images.shape[2] < self.input_mask.shape[1]:
            raise ValueError('Images too small.')

        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)
        noisy_inputs, noisy_outputs = self._preprocess(noisy_images)
        
        print 'train stat: input shape', noisy_inputs.shape, 'output shape:', outputs.shape
        
        if precondition:
            if self.verbosity > 0:
                print 'Preconditioning...'

            # remove correlations
            noisy_inputs, outputs = self._precondition(noisy_inputs, outputs)

        # indicates which layers will be trained
        train_layers = [self.num_layers - 1] if train_top_layer else range(self.num_layers)

        if self.verbosity > 0:
            print 'Creating SLSTMs...'

        # create SLSTMs
        for l in range(self.num_layers):
            self.slstm[l] = SLSTM(
                num_rows=inputs.shape[1],
                num_cols=inputs.shape[2],
                num_channels=inputs.shape[3] if l < 1 else self.num_hiddens,
                num_hiddens=self.num_hiddens,
                batch_size=min([batch_size, self.MAX_BATCH_SIZE]),
                nonlinearity=self.nonlinearity,
                extended=self.extended,
                slstm=self.slstm[l],
                verbosity=self.verbosity)
            print 'done creating SLSTM'
        # compute loss function and its gradient
        def f_df(params, idx):
            # set model parameters
            for l in train_layers:
                self.slstm[l].set_parameters(params['slstm'][l])
            self.mcgsm._set_parameters(params['mcgsm'], {'train_means': train_means})

            # select batch and compute hidden activations
            Y = outputs[idx:idx + batch_size]
            H = noisy_inputs[idx:idx + batch_size]

            for l in range(self.num_layers):
                H = self.slstm[l].forward(H)

            # form inputs to MCGSM
            H_flat = H.reshape(-1, self.num_hiddens).T
            Y_flat = Y.reshape(-1, self.num_channels).T

            norm_const = -H_flat.shape[1]

            # compute gradients
            df_dh, _, loglik = self.mcgsm._data_gradient(H_flat, Y_flat)
            df_dh = df_dh.T.reshape(*H.shape) / norm_const

            # average log-likelihood
            # print 'loglik', loglik
            # print sum(loglik) / H_flat.shape[1]
            f = sum(loglik) / norm_const
            
            df_dtheta = {}
            df_dtheta['slstm'] = [0.] * self.num_layers

            for l in range(self.num_layers)[::-1]:
                if l not in train_layers:
                    break
                if l > min(train_layers):
                    # derivative with respect to inputs of layer l are derivatives
                    # of hidden states of layer l - 1
                    df_dtheta['slstm'][l] = self.slstm[l].backward(df_dh, force_backward=True)
                    df_dh = df_dtheta['slstm'][l]['inputs']
                    del df_dtheta['slstm'][l]['inputs']

                else:
                    # no need to compute derivatives with respect to input units
                    df_dtheta['slstm'][l] = self.slstm[l].backward(df_dh)

            # compute gradient of MCGSM
            df_dtheta['mcgsm'] = self.mcgsm._parameter_gradient(H_flat, Y_flat,
                parameters={'train_means': train_means}) * log(2.) * self.mcgsm.dim_out

            return f, df_dtheta

        # collect current parameters
        params = {}
        params['slstm'] = [0.] * self.num_layers
        for l in range(self.num_layers)[::-1]:
            if l not in train_layers:
                break
            params['slstm'][l] = self.slstm[l].parameters()
        params['mcgsm'] = self.mcgsm._parameters({'train_means': train_means})

        # a start index for each batch
        start_indices = range(
            0, inputs.shape[0] - batch_size + 1, batch_size)

        if self.verbosity > 0:
            print 'Training...'

        if method.upper() == 'SFO':
            try:
                # optimize using sum-of-functions optimizer
                optimizer = SFO(f_df, params, start_indices, display=self.verbosity)
                params_opt = optimizer.optimize(num_passes=num_epochs)

                # set model parameters
                for l in range(self.num_layers):
                    self.slstm[l].set_parameters(params_opt['slstm'][l])
                self.mcgsm._set_parameters(params_opt['mcgsm'], {'train_means': train_means})

            except KeyboardInterrupt:
                pass

            return optimizer.hist_f_flat

        elif method.upper() == 'SGD':
            loss = []
            diff = {
                'slstm': [0.] * self.num_layers,
                'mcgsm': zeros_like(params['mcgsm'])}

            for l in train_layers:
                diff['slstm'][l] = {}
                for key in params['slstm'][l]:
                    if save_grad is not None:
                        #print 'creating grad txts'
                        fname = grad_fname+'slstm'+str(l)+'_'+key
                        pname = grad_fname+'slstm'+str(l)+'_para_'+key
                        #if(not os.path.isfile(fname+'.txt')):
                        g = open(fname+'.txt','w')
                        g.close()
                        p = open(pname+'.txt','w')
                        p.close()
                    diff['slstm'][l][key] = zeros_like(params['slstm'][l][key])

            for n in range(num_epochs):
                for b in range(0, inputs.shape[0] - batch_size + 1, batch_size):
                    # compute gradients
                    f, df = f_df(params, b)
                    
                    loss.append(f / log(2.) / self.num_channels)


                    # update SLSTM parameters
                    for l in train_layers:
                        for key in params['slstm'][l]:
                            diff['slstm'][l][key] = momentum * diff['slstm'][l][key] - df['slstm'][l][key]
                            #print type(diff['slstm'][l][key]),diff['slstm'][l][key].shape
                            if save_grad is not None:
                                fname = grad_fname+'slstm'+str(l)+'_'+key
                                #print fname
                                f = open(fname+'.txt','a')
                                l2_norm = norm(diff['slstm'][l][key])
                                f.write(str(l2_norm)+'\n')
                                f.close()
                            params['slstm'][l][key] = params['slstm'][l][key] + learning_rate * diff['slstm'][l][key]
                            #save param norms
                            pname = grad_fname+'slstm'+str(l)+'_para_'+key
                            p = open(pname+'.txt','a')
                            l2_norm = norm(params['slstm'][l][key])
                            p.write(str(l2_norm)+'\n')
                            p.close()

                    # update MCGSM parameters
                    diff['mcgsm'] = momentum * diff['mcgsm'] - df['mcgsm']
                    params['mcgsm'] = params['mcgsm'] + learning_rate * diff['mcgsm']

                    if self.verbosity > 0:
                        print '{0:>5} {1:>10.4f} {2:>10.4f}'.format(
                            n, loss[-1], mean(loss[-max([10, 20000 // batch_size]):]))
            return loss

        elif method.upper() == 'ADAM':
            loss = []
            diff_mean = {
                'slstm': [0.] * self.num_layers,
                'mcgsm': zeros_like(params['mcgsm'])}
            diff_sqrd = {
                'slstm': [0.] * self.num_layers,
                'mcgsm': zeros_like(params['mcgsm'])}

            for l in train_layers:
                diff_mean['slstm'][l] = {}
                diff_sqrd['slstm'][l] = {}
                for key in params['slstm'][l]:
                    diff_mean['slstm'][l][key] =    zeros_like(params['slstm'][l][key])
                    diff_sqrd['slstm'][l][key] = zeros_like(params['slstm'][l][key])

            # step counter
            t = 1

            for n in range(num_epochs):
                for b in range(0, inputs.shape[0] - batch_size + 1, batch_size):
                    # compute gradients
                    f, df = f_df(params, b)

                    loss.append(f / log(2.) / self.num_channels)

                    # include bias correction in step width
                    step_width = learning_rate / (1. - power(decay1, t)) * sqrt(1. - power(decay2, t))
                    t += 1

                    # update SLSTM parameters
                    for l in train_layers:
                        for key in params['slstm'][l]:
                            diff_mean['slstm'][l][key] = decay1 * diff_mean['slstm'][l][key] \
                                + (1. - decay1) * df['slstm'][l][key]
                            diff_sqrd['slstm'][l][key] = decay2 * diff_sqrd['slstm'][l][key] \
                                + (1. - decay2) * square(df['slstm'][l][key])

                            params['slstm'][l][key] = params['slstm'][l][key] - \
                                step_width * diff_mean['slstm'][l][key] / (1e-8 + sqrt(diff_sqrd['slstm'][l][key]))

                    # update MCGSM parameters
                    diff_mean['mcgsm'] = decay1 * diff_mean['mcgsm'] + (1. - decay1) * df['mcgsm']
                    diff_sqrd['mcgsm'] = decay2 * diff_sqrd['mcgsm'] + (1. - decay2) * square(df['mcgsm'])
                    params['mcgsm'] = params['mcgsm'] - \
                        step_width * diff_mean['mcgsm'] / (1e-8 + sqrt(diff_sqrd['mcgsm']))

                    if self.verbosity > 0:
                        print '{0:>5} {1:>10.4f} {2:>10.4f}'.format(
                            n,
                            loss[-1],
                            mean(loss[-max([10, 20000 // batch_size]):]))

            return loss

        else:
            raise ValueError('Unknown method \'{0}\'.'.format(method))




    def finetune(self, images,
        max_iter=1000,
        train_means=False,
        num_samples_train=500000,
        num_samples_valid=100000,
        err_flag = 1):
        """
        Train MCGSM using L-BFGS while keeping parameters of spatial LSTMs fixed.

        @type  images: C{ndarray}/C{list}
        @param images: an array or a list of images

        @type  max_iter: C{int}
        @param max_iter: maximum number of L-BFGS iterations

        @type  train_means: C{bool}
        @param train_means: whether or not to optimize the mean parameters of the MCGSM

        @type  num_samples_train: C{int}
        @param num_samples_train: number of training examples extracted from images

        @type  num_samples_valid: C{int}
        @type  num_samples_valid: number of validation examples used for early stopping

        @rtype: C{bool}
        @return: true if training converged, false otherwise
        """

        if images.shape[0] > min([200000, num_samples_train]):
            images = images[random_select(min([200000, num_samples_train]), images.shape[0])]

        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)

        if self.preconditioner:
            if self.verbosity > 0:
                print 'Preconditioning...'

            # remove correlations
            inputs, outputs = self._precondition(inputs, outputs)

        print 'After preprocessing and preconditioning'
        print 'input size',inputs.shape
        print 'output size',outputs.shape

        # compute hidden unit activations
        hiddens = inputs

        if self.verbosity > 0:
            print 'Computing hidden states...'

        for l in range(self.num_layers):
            print 'Creating SLSTM'
            self.slstm[l] = SLSTM(
                num_rows=hiddens.shape[1],
                num_cols=hiddens.shape[2],
                num_channels=hiddens.shape[3],
                num_hiddens=self.num_hiddens,
                batch_size=min([hiddens.shape[0], 32]), #Change: self.MAX_BATCH_SIZE = 32
                nonlinearity=self.nonlinearity,
                extended=self.extended,
                slstm=self.slstm[l],
                verbosity=self.verbosity)

            print 'Forward '
            hiddens = self.slstm[l].forward(hiddens)
            print 'Hiddens for MCGSM shape', hiddens.shape
            print 'Outputs shape',outputs.shape

        if self.verbosity > 0:
            print 'Preparing inputs and outputs...'

        # form inputs to MCGSM
        hiddens = hiddens.reshape(-1, self.num_hiddens).T
        outputs = outputs.reshape(-1, self.num_channels).T

        if hiddens.shape[1] > num_samples_train:
            num_samples_valid = min([num_samples_valid, hiddens.shape[1] - num_samples_train])

            print 'Doing validation with', num_samples_valid,'samples'
            # select subset of data points for finetuning
            idx = random_select(num_samples_train + num_samples_valid, hiddens.shape[1])

            if num_samples_valid > 0:
                # split data into training and validation set
                hiddens_train = asarray(hiddens[:, idx[:num_samples_train]], order='F')
                outputs_train = asarray(outputs[:, idx[:num_samples_train]], order='F')
                hiddens_valid = asarray(hiddens[:, idx[num_samples_train:]], order='F')
                outputs_valid = asarray(outputs[:, idx[num_samples_train:]], order='F')

                # finetune with early stopping based on validation performance
                return self.mcgsm.train(
                    hiddens_train, outputs_train,
                    hiddens_valid, outputs_valid,
                    parameters={
                        'verbosity': self.verbosity,
                        'train_means': train_means,
                        'max_iter': max_iter})

            else:
                hiddens = asarray(hiddens[:, idx], order='F')
                outputs = asarray(outputs[:, idx], order='F')

        if self.verbosity > 0:
            print 'Finetuning...'

        return self.mcgsm.train(hiddens, outputs, parameters={
            'verbosity': self.verbosity,
            'train_means': train_means,
            'max_iter': max_iter})

    def finetune_noisy(self, images,noisy_images,
        max_iter=1000,
        train_means=False,
        num_samples_train=500000,
        num_samples_valid=100000,
        err_flag = 1):
        """
        Train MCGSM using L-BFGS while keeping parameters of spatial LSTMs fixed.

        @type  images: C{ndarray}/C{list}
        @param images: an array or a list of images

        @type  max_iter: C{int}
        @param max_iter: maximum number of L-BFGS iterations

        @type  train_means: C{bool}
        @param train_means: whether or not to optimize the mean parameters of the MCGSM

        @type  num_samples_train: C{int}
        @param num_samples_train: number of training examples extracted from images

        @type  num_samples_valid: C{int}
        @type  num_samples_valid: number of validation examples used for early stopping

        @rtype: C{bool}
        @return: true if training converged, false otherwise
        """

        if images.shape[0] > min([200000, num_samples_train]):
            r_s = random_select(min([200000, num_samples_train]), images.shape[0])
            images = images[r_s]
            noisy_images = noisy_images[r_s]
            
        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)
        noisy_inputs,noisy_outputs = self._preprocess(noisy_images)

        if self.preconditioner:
            if self.verbosity > 0:
                print 'Preconditioning...'

            print 'difference',(inputs-noisy_inputs).sum()
            print 'difference',(outputs-noisy_outputs).sum()
            # remove correlations
            noisy_inputs, outputs = self._precondition(noisy_inputs, outputs)

        print 'After preprocessing and preconditioning'
        print 'input size',noisy_inputs.shape
        print 'output size',outputs.shape

        # compute hidden unit activations
        hiddens = noisy_inputs

        if self.verbosity > 0:
            print 'Computing hidden states...'

        for l in range(self.num_layers):
            print 'Creating SLSTM'
            self.slstm[l] = SLSTM(
                num_rows=hiddens.shape[1],
                num_cols=hiddens.shape[2],
                num_channels=hiddens.shape[3],
                num_hiddens=self.num_hiddens,
                batch_size=min([hiddens.shape[0], 32]), #Change: self.MAX_BATCH_SIZE = 32
                nonlinearity=self.nonlinearity,
                extended=self.extended,
                slstm=self.slstm[l],
                verbosity=self.verbosity)

            print 'Forward '
            hiddens = self.slstm[l].forward(hiddens)
            print 'Hiddens for MCGSM shape', hiddens.shape
            print 'Outputs shape',outputs.shape

        if self.verbosity > 0:
            print 'Preparing inputs and outputs...'

        # form inputs to MCGSM
        hiddens = hiddens.reshape(-1, self.num_hiddens).T
        outputs = outputs.reshape(-1, self.num_channels).T

        if hiddens.shape[1] > num_samples_train:
            num_samples_valid = min([num_samples_valid, hiddens.shape[1] - num_samples_train])

            print 'Doing validation with', num_samples_valid,'samples'
            # select subset of data points for finetuning
            idx = random_select(num_samples_train + num_samples_valid, hiddens.shape[1])

            if num_samples_valid > 0:
                # split data into training and validation set
                hiddens_train = asarray(hiddens[:, idx[:num_samples_train]], order='F')
                outputs_train = asarray(outputs[:, idx[:num_samples_train]], order='F')
                hiddens_valid = asarray(hiddens[:, idx[num_samples_train:]], order='F')
                outputs_valid = asarray(outputs[:, idx[num_samples_train:]], order='F')

                # finetune with early stopping based on validation performance
                return self.mcgsm.train(
                    hiddens_train, outputs_train,
                    hiddens_valid, outputs_valid,
                    parameters={
                        'verbosity': self.verbosity,
                        'train_means': train_means,
                        'max_iter': max_iter,
                        'train_means' : True,
                        'train_linear_features' : True})

            else:
                hiddens = asarray(hiddens[:, idx], order='F')
                outputs = asarray(outputs[:, idx], order='F')

        if self.verbosity > 0:
            print 'Finetuning...'

        return self.mcgsm.train(hiddens, outputs, parameters={
            'verbosity': self.verbosity,
            'train_means': train_means,
            'max_iter': max_iter})

    def hidden_states(self, images, return_all=False, layer=None):
        """
        Compute hidden states of LSTM units for given images.

        By default, the last layer's hidden units are computed.

        @type  images: C{ndarray}/C{list}
        @param images: array or list of images to process

        @type  return_all: C{bool}
        @param return_all: if true, also return preconditioned inputs and outputs

        @type  layer: C{int}
        @param layer: a positive integer controlling which layer's hidden units to compute

        @rtype: C{ndarray}/C{tuple}
        @return: hidden states or a tuple of inputs, hidden states, and outputs
        """

        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)

        if self.preconditioner is not None:
            if self.verbosity > 0:
                print 'Preconditioning...'

            # remove correlations
            inputs, outputs = self._precondition(inputs, outputs)

        # compute hidden unit activations
        hiddens = inputs

        batch_size = min([hiddens.shape[0], self.MAX_BATCH_SIZE])

        if layer is None or layer < 1:
            layer = self.num_layers

        for l in range(layer):
            if self.slstm[l].num_rows != hiddens.shape[1] \
                or self.slstm[l].num_cols != hiddens.shape[2] \
                or self.slstm[l].batch_size != batch_size:
                self.slstm[l] = SLSTM(
                    num_rows=hiddens.shape[1],
                    num_cols=hiddens.shape[2],
                    num_channels=hiddens.shape[3],
                    num_hiddens=self.num_hiddens,
                    batch_size=batch_size,
                    nonlinearity=self.nonlinearity,
                    extended=self.extended,
                    slstm=self.slstm[l],
                    verbosity=self.verbosity)

            hiddens = self.slstm[l].forward(hiddens)

        if return_all:
            return inputs, hiddens, outputs
        return hiddens



    def gradient(self, images, precond = None,niter=0,path=None,ent_max = 4.0):
        """
        Returns the average log-likelihood [nat] and its gradient with respect to pixel values.

        @type  images: C{ndarray}
        @param images: images at which to evaluate the density's gradient

        @rtype: C{tuple}
        @return: average log-likelihood and gradient with respect to images
        """

        self.verbosity =0
        inputs, outputs = self._preprocess(images)

        if self.preconditioner:
            inputs, outputs = self._precondition(inputs, outputs)

        # create SLSTMs
        batch_size = min([images.shape[0], self.MAX_BATCH_SIZE])
        for l in range(self.num_layers):
            if self.slstm[l] is None or \
                self.slstm[l].batch_size != batch_size or \
                self.slstm[l].num_rows != inputs.shape[1] or \
                self.slstm[l].num_cols != inputs.shape[2]:
                self.slstm[l] = SLSTM(
                    num_rows=inputs.shape[1],
                    num_cols=inputs.shape[2],
                    num_channels=inputs.shape[3] if l < 1 else self.num_hiddens,
                    num_hiddens=self.num_hiddens,
                    batch_size=batch_size,
                    nonlinearity=self.nonlinearity,
                    extended=self.extended,
                    slstm=self.slstm[l],
                    verbosity=self.verbosity)

        # compute hidden unit activations
        hiddens = inputs
        for l in range(self.num_layers):
            hiddens = self.slstm[l].forward(hiddens)
            if l == 0:
                hiddens_1 = hiddens

        # form inputs to MCGSM
        H_flat = hiddens.reshape(-1, self.num_hiddens).T
        Y_flat = outputs.reshape(-1, self.num_channels).T
        #print 'outputs shape',outputs.shape

        # compute gradients
        df_dh, df_dy, loglik = self.mcgsm._data_gradient(H_flat, Y_flat)
        df_dh = df_dh.T.reshape(*hiddens.shape) / H_flat.shape[1]
        df_dy = df_dy.T.reshape(*outputs.shape) / H_flat.shape[1]

        # average log-likelihood
        f = loglik/ H_flat.shape[1]
        f = f.reshape(outputs.shape[0],outputs.shape[1],outputs.shape[2],1) #Check
        for l in range(self.num_layers)[::-1]:
            df_dh = self.slstm[l].backward(df_dh, force_backward=True)['inputs']

        # compute posterior
        posterior = self.mcgsm.posterior(H_flat, Y_flat)                
        log_posterior = np.log2(posterior)
        entropy = posterior*log_posterior
        entropy[np.isnan(entropy)] = 0
        entropy = entropy.sum(axis=0)        
        entropy = -entropy.T.reshape(outputs.shape[0],outputs.shape[1],outputs.shape[2])
        _l = np.zeros((images.shape[0],images.shape[1],images.shape[2])) 
        _l[:,2:,2:-2] = np.copy(entropy)
        entropy = _l   

        if self.preconditioner:
            df_dh, df_dy = self._adjust_gradient(df_dh, df_dy)

        # locate output pixel in output mask
        for i_off, j_off in zip(
                range(self.output_mask.shape[0]),
                range(self.output_mask.shape[1])):
            if any(self.output_mask[i_off, j_off]):
                break

        gradient = zeros_like(images)
        whitened_images = copy(images)
        max_grad = 10000*np.zeros_like(images)
        # make sure mask and gradient have compatible dimensionality
        if gradient.ndim == 4 and self.input_mask.ndim == 2:
            gradient = gradient[:, :, :, 0]
            whitened_images = whitened_images[:,:,:,0]            

        f_arr = np.zeros((images.shape[0],images.shape[1],images.shape[2],1))

        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                patch3 = whitened_images[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                patch3[:, self.output_mask] = outputs[:,i,j]
                f_arr[:,i+i_off,j+j_off,:] = f[:,i,j]

                idxs = entropy[:,i+i_off,j+j_off] < ent_max
                idxs = [idx for idx,b in enumerate(idxs) if b]
                tmp_mask = np.zeros((images.shape[0],1))
                tmp_mask[idxs] = 1
                #idxs = np.array(idxs)
                #print idxs
                #print len(idxs)
                #if entropy[i+i_off,j+j_off] < ent_max :
                patch = gradient[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                patch[:, self.input_mask] += np.multiply(tmp_mask,df_dh[:,i,j])
                patch[:, self.output_mask] += np.multiply(tmp_mask,df_dy[:,i,j])

        #For Saving Gradient                
        # if (niter%10 ==0):
        #     for k in range(images.shape[0]):
        #         fig1 = plt.figure(1)
        #         plt.imshow(entropy[k,:,:].squeeze(),vmin=0.0,vmax=5.0)
        #         plt.colorbar()
        #         plt.savefig(path+str(k+1)+'/entropy_img'+str(niter))
        #         plt.close(fig1)

        return f_arr,gradient.reshape(*images.shape), whitened_images.reshape(*images.shape)


    def get_image(self, inputs, outputs, img_shape, grad=True):
        if not grad:
            inputs, outputs = self._precondition_inverse(inputs, outputs)
        
        gradient = np.zeros(img_shape)

        # make sure mask and gradient have compatible dimensionality
        if gradient.ndim == 4 and self.input_mask.ndim == 2:
            gradient = gradient[:, :, :, 0]

        count = np.zeros(gradient.shape)
        for i in range(img_shape[1] - self.input_mask.shape[0] + 1):
            for j in range(img_shape[2] - self.input_mask.shape[1] + 1):
                patch = gradient[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                patch2 = count[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                
                patch[:, self.input_mask] += inputs[:, i, j]
                patch2[:, self.input_mask] += 1
                patch[:, self.output_mask] += outputs[:, i, j]
                patch2[:, self.output_mask] += 1
                
        count[count==0] = 1
        gradient = np.multiply(gradient,1/count.astype(float))
        if(any(count==0)):
            print 'ZERO COUNT'
        #print count
        return gradient.reshape(*img_shape)

    def sample(self,
            images,
            min_values=None,
            max_values=None,
            mask=None,
            return_loglik=False):
        """
        Sample one or several images.

        @type  images: C{ndarray}/C{list}
        @param images: an array or a list of images to initialize pixels at boundaries

        @type  min_values: C{ndarray}/C{list}
        @param min_values: list of lower bounds for each channel (for increased stability)

        @type  max_values: C{ndarray}/C{list}
        @param max_values: list of upper bounds for each channel (for increased stability)

        @type  mask: C{ndarray}
        @param mask: replace only certain pixels indicated by this Boolean mask

        @rtype: C{ndarray}
        @return: sampled images of the size of the images given as input
        """

        # reshape images into four-dimensional arrays
        shape = images.shape
        if images.ndim == 2:
            images = images[None, :, :, None]
        elif images.ndim == 3:
            if self.num_channels > 1:
                images = images[None]
            else:
                images = images[:, :, :, None]

        # create spatial LSTMs for sampling
        print 'creating lstm'
        for l in range(self.num_layers):
            if self.slstm[l].num_rows != 1 \
                or self.slstm[l].num_cols != 1 \
                or self.slstm[l].batch_size != images.shape[0]:
                self.slstm[l] = SLSTM(
                    num_rows=1,
                    num_cols=1,
                    num_channels=sum(self.input_mask) if l < 1 else self.num_hiddens,
                    num_hiddens=self.num_hiddens,
                    batch_size=images.shape[0],
                    nonlinearity=self.nonlinearity,
                    slstm=self.slstm[l],
                    extended=self.extended)

        # container for hidden and memory unit activations
        hiddens = []
        memory = []
        for l in range(self.num_layers):
            hiddens.append(defaultdict(lambda: 0.))
            memory.append(defaultdict(lambda: 0.))

        #print self.output_mask.shape, self.output_mask
        #print self.input_mask.shape, self.input_mask
        # locate output pixel
        for i_off, j_off in zip(
                range(self.output_mask.shape[0]),
                range(self.output_mask.shape[1])):
            if any(self.output_mask[i_off, j_off]):
                break
        #print i_off, j_off
        if min_values is not None:
            min_values = asarray(min_values).reshape(1, 1, 1, -1)
            if self.output_mask.ndim > 2:
                min_values = min_values[:, :, :, self.output_mask[i_off, j_off]]
        if max_values is not None:
            max_values = asarray(max_values).reshape(1, 1, 1, -1)
            if self.output_mask.ndim > 2:
                max_values = max_values[:, :, :, self.output_mask[i_off, j_off]]

        # unnormalized log-density of generated sample
        logq = 0.

        h_norm =  zeros([images.shape[1] - self.input_mask.    shape[0] + 1,images.shape[2] - self.input_mask.shape[1] + 1])
        c_norm =  zeros([images.shape[1] - self.input_mask.shape[0] + 1,images.shape[2] - self.input_mask.shape[1] + 1])
        tanh_c =  zeros([images.shape[1] - self.input_mask.shape[0] + 1,images.shape[2] - self.input_mask.shape[1] + 1])

        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            print i
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                # extract patches from images
                # print 'patch extraction'
                patches = images[:,
                    i:i + self.input_mask.shape[0],
                    j:j + self.input_mask.shape[1]]

                # extract causal neighborhoods from patches
                inputs = []
                for k in range(images.shape[0]):
                    inputs.append(
                        generate_data_from_image(
                            patches[k, :, :], self.input_mask, self.output_mask)[0])
                inputs = asarray(inputs)
                inputs = inputs.reshape(inputs.shape[0], 1, 1, -1)

                if self.preconditioner:
                    inputs = self._precondition(inputs)

                # set hidden unit activations
                for l in range(self.num_layers):
                    self.slstm[l].net.blobs['h_init_i_jm1'].data[:] = hiddens[l][i, j - 1]
                    self.slstm[l].net.blobs['h_init_im1_j'].data[:] = hiddens[l][i - 1, j]
                    self.slstm[l].net.blobs['c_init_i_jm1'].data[:] = memory[l][i, j - 1]
                    self.slstm[l].net.blobs['c_init_im1_j'].data[:] = memory[l][i - 1, j]

                # compute hidden unit activations
                # print 'hidden activation'
                activations = inputs

                for l in range(self.num_layers):
                    activations = self.slstm[l].forward(activations)

                # store hidden unit activations
                for l in range(self.num_layers):
                    hiddens[l][i, j] = self.slstm[l].net.blobs['outputs'].data.copy()
                    memory[l][i, j] = self.slstm[l].net.blobs['c_0_0'].data.copy()


                h_norm[i,j] = norm(hiddens[-1][i,j])
                c_norm[i,j] = norm(memory[-1][i,j])
                tanh_c[i,j] = sum(sum(tanh(memory[-1][i,j])))

                if mask is not None and not mask[i + i_off, j + j_off].all():
                    # skip sampling of this pixel
                    continue

                for _ in range(10):
                    # sample MCGSM
                    # print 'Hiddens shape',hiddens[-1][i, j].reshape(-1, self.num_hiddens).T.shape

                    # print 'Inputs shape',inputs.reshape(-1, 12).T.shape

                    outputs = self.mcgsm.sample(
                        hiddens[-1][i, j].reshape(-1, self.num_hiddens).T)

                    # outputs = self.mcgsm.sample(
                        # inputs.reshape(-1, 12).T)
                    
                    if not any(isnan(outputs)):
                        break
                    print 'Warning: NaNs detected.'

                #print outputs
                if return_loglik:
                    logq += self.mcgsm.loglikelihood(
                        hiddens[-1][i, j].reshape(-1, self.num_hiddens).T,
                        outputs)
                
                outputs = outputs.T.reshape(outputs.shape[1], 1, 1, outputs.shape[0])


                if self.preconditioner:
                    inputs, outputs = self._precondition_inverse(inputs, outputs)

                if max_values is not None:
                    outputs[outputs > max_values] = max_values[outputs > max_values]
                if min_values is not None:
                    outputs[outputs < min_values] = min_values[outputs < min_values]

                # insert sampled pixels into images
                if self.output_mask.ndim > 2:
                    images[:, i + i_off, j + j_off][:, self.output_mask[i_off, j_off]] = outputs
                else:
                    images[:, i + i_off, j + j_off] = outputs

        images = images.reshape(*shape)

        sample_act = {'h':h_norm,'c':c_norm,'t':tanh_c}
        savemat('sample_act.mat',sample_act)

        if return_loglik:
            return images, logq
        return images

    def _logq(self, images, mask):
        """
        Computes an unnormalized conditional log-likelihood used by Metropolis-Hastings (e.g., for inpainting).
        """

        inputs, hiddens, outputs = self.hidden_states(images, return_all=True)

        # locate output pixel
        for i_off, j_off in zip(
                range(self.output_mask.shape[0]),
                range(self.output_mask.shape[1])):
            if any(self.output_mask[i_off, j_off]):
                break

        # unnormalized log-density of generated sample
        logq = 0.

        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                if not mask[i + i_off, j + j_off]:
                    # skip evaluation of this pixel
                    continue

                logq += self.mcgsm.loglikelihood(
                    hiddens[:, i, j, :].reshape(-1, self.num_hiddens).T,
                    outputs[:, i, j, :])

        return logq



    def __setstate__(self, state):
        """
        Method used by pickle module, for backwards compatibility reasons.
        """

        self.__dict__ = state

        if not hasattr(self, 'nonlinearity'):
            self.nonlinearity = 'TanH'
        if not hasattr(self, 'extended'):
            self.extended = False
