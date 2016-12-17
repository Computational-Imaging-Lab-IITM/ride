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



    def gradient(self, images, precond = None,niter=0,path=None,ent_max = 4.0,mask=None):
        """
        Returns the average log-likelihood [nat] and its gradient with respect to pixel values.

        @type  images: C{ndarray}
        @param images: images at which to evaluate the density's gradient

        @rtype: C{tuple}
        @return: average log-likelihood and gradient with respect to images
        """
        #Check

        sigma_s =15
        self.verbosity =0
        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)

        #if self.preconditioner:
        if self.verbosity > 0:
       
            print 'Preconditioning...'

        #self.preconditioner = None
        inputs_1 = copy(inputs)
        outputs_1 = copy(outputs)
        # remove correlations

        inputs_1, outputs_1 = self._precondition(inputs_1, outputs_1)
        if precond is not None :
            if self.preconditioner.pre_out > precond.pre_out:
                #print 'In condition',self.preconditioner.pre_out
                print 'Increased'
                #self.preconditioner = precond          
                #print 'trained pre_out',self.preconditioner.pre_out

        #self.preconditioner = None
        if self.preconditioner:
            inputs, outputs = self._precondition(inputs, outputs)

        # inputs = np.zeros_like(inputs) #CHANGE
        # outputs = np.zeros_like(outputs)

        if self.verbosity > 0:
            print 'Creating SLSTMs...'

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
        #print 'intial df_dh shape',df_dh.shape, 'intial df_dy shape',df_dy.shape
        df_dh = df_dh.T.reshape(*hiddens.shape) / H_flat.shape[1]
        df_dy = df_dy.T.reshape(*outputs.shape) / H_flat.shape[1]
        df_dh_w = df_dh
        df_dy_w = df_dy

        # average log-likelihood


        f = loglik/ H_flat.shape[1]
        #print 'f',f.sum()
        f = f.reshape(1,outputs.shape[1],outputs.shape[2],1) #Checl
        for l in range(self.num_layers)[::-1]:
            df_dh = self.slstm[l].backward(df_dh, force_backward=True)['inputs']

        # compute posterior
        posterior = self.mcgsm.posterior(H_flat, Y_flat)                
        #print posterior.sum(0).sum(), posterior.min(), posterior.max()
        log_posterior = np.log2(posterior)
        entropy = posterior*log_posterior
        
        entropy[np.isnan(entropy)] = 0
        entropy = entropy.sum(axis=0)        
        entropy = -entropy.T.reshape(outputs.shape[1],outputs.shape[2])
        # print 'sum', sum(sum(np.isnan(entropy)))
        # print 'entropy', entropy.min(), entropy.max()
        # print entropy.shape


        _l = np.zeros((images.shape[1],images.shape[2])) 
        _l[2:,2:-2] = np.copy(entropy)
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
        #gradient_w = zeros((images.shape[1],images.shape[2],self.num_hiddens))
        whitened_images = copy(images)
        max_grad = 10000*np.zeros_like(images)
        # make sure mask and gradient have compatible dimensionality
        if gradient.ndim == 4 and self.input_mask.ndim == 2:
            gradient = gradient[:, :, :, 0]
            #gradient_w = gradient_w[:,:,:,0]

        #if self.input_mask.ndim !=2:
        #    self.input_mask[:,:,0] = self.input_mask[:,:,-1]

        count = np.zeros(gradient.shape)
        #img = images.reshape(images.shape[1],images.shape[2])
        f_arr = np.zeros((1,images.shape[1],images.shape[2],1))
        #print 'input_mask shape',self.input_mask
        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                #ipm = copy(self.input_mask)
                patch = gradient[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                patch2 = count[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                patch3 = whitened_images[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]

                #ipm[patch2[0,:,:]>=2] = False
                patch2[:, self.input_mask] += 1
                patch2[:, self.output_mask] += 1
                patch3[:, self.output_mask] = outputs[:,i,j,:]
                f_arr[:,i+i_off,j+j_off,:] = f[:,i,j,:]

                # if i in [3,5] and j in [3,5]:
                    # print i,j
                    # print 'Input before',patch[:,self.input_mask]
                    # print 'df_dh',df_dh[:,i,j]
                # dfdh = df_dh[:,i,j]
                #print dfdh
                #print ipm.reshape(1,-1)[:,:12]
                #patch[:, self.input_mask] += dfdh*ipm.reshape(1,-1)[:,:12]
                #print 'dfdh',df_dh[:,i,j].shape
                if entropy[i+i_off,j+j_off] < ent_max  :
                    patch = gradient[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                    patch[:, self.input_mask] += df_dh[:,i,j]
                    patch[:, self.output_mask] += df_dy[:,i,j]
                
                #gradient_w[i+i_off,j+j_off,:] = df_dy_w[:,i,j]
                # if 1:
                #     print 'i','j',i,j
                #     print 'df_dh',df_dh[:,i,j]
                #     print 'df_dy',df_dy[:,i,j]
                #     print gradient[:,i,j]
                # #Bilateral Filtering
                # input_weights = np.exp(-np.abs(img[self.input_mask]-img[self.output_mask]*(12))**2)/(2*sigma_s**2)
                # #print 'input weights shape',input_weights.shape
                # patch[:,self.input_mask] += input_weights*df_dh[:,i,j] 
                # output_weight = np.exp(-np.abs(np.mean(img[self.input_mask])-img[self.output_mask])**2)/(2*sigma_s**2)
                # patch[:,self.output_mask] += output_weight*df_dy[:,i,j]
                

                # patches_inp = patch3[:,self.input_mask].flatten() 
                # dfdh = df_dh[:,i,j].flatten()
                # #print dfdh.shape
                # #print patch3[:,self.input_mask].shape
                # for k,l1 in enumerate(patches_inp):
                #     patches_inp[k] = l1 if abs(l1) > abs(dfdh[k]) else dfdh[k]
                # patch3[:,self.input_mask] = patches_inp.reshape(*patch3[:,self.input_mask].shape)   
                # patches_op = patch3[:,self.output_mask]
                # patch3[:,self.output_mask] = patches_op if np.abs(patches_op) > np.abs(df_dy[:,i,j]) else df_dy[:,i,j]

                # if i in [a3,5] and j in [3,5]:    
                    # print 'Input after',patch[:,self.input_mask]
        count[count==0] = 1
        #gradient = np.multiply(gradient,1.0/count.astype(float))
        #print 'gradient shape',gradient.shape
        #print 'Hidden grad', np.abs(gradient_w).sum()
        #print inputs.shape

        # for l in range(self.num_layers):
        #     print'h_init_i_jm1', self.slstm[l].net.blobs['h_init_i_jm1'].data[:] 
        #     print 'h_init_im1_j',self.slstm[l].net.blobs['h_init_im1_j'].data[:]
        #     print 'c_init_i_jm1',self.slstm[l].net.blobs['c_init_i_jm1'].data[:]
        #     print 'c_init_im1_j',self.slstm[l].net.blobs['c_init_im1_j'].data[:]


        inputs_norm = np.linalg.norm(inputs.reshape(inputs.shape[1],inputs.shape[2],-1),axis = 2)
        #grad_h_norm = np.linalg.norm(gradient_w,axis=2)
        hiddens_1_norm = np.linalg.norm(hiddens_1.reshape(hiddens_1.shape[1],hiddens_1.shape[2],-1),axis = 2)
        hiddens_norm = np.linalg.norm(hiddens.reshape(hiddens.shape[1],hiddens.shape[2],-1),axis = 2)
        
        # fig5 = plt.figure(5)
        # xx,yy = np.mgrid[0:hiddens_norm.shape[0],0:hiddens_norm.shape[1]]
        # ax = fig5.gca(projection='3d')
        # ax.plot_surface(xx,yy,inputs_norm,cmap=cm.coolwarm,linewidth=0)
        # plt.show()

        # fig6 = plt.figure(6)
        # xx,yy = np.mgrid[0:hiddens_1_norm.shape[0],0:hiddens_1_norm.shape[1]]
        # ax = fig6.gca(projection='3d')
        # ax.plot_surface(xx,yy,hiddens_1_norm,cmap=cm.coolwarm,linewidth=0)
        # plt.show()

        # fig7 = plt.figure(7)
        # xx,yy = np.mgrid[0:hiddens_norm.shape[0],0:hiddens_norm.shape[1]]
        # ax = fig7.gca(projection='3d')
        # ax.plot_surface(xx,yy,hiddens_norm,cmap=cm.coolwarm,linewidth=0)
        # plt.show()

        # fig8 = plt.figure(8)
        # plt.imshow(hiddens_norm,cmap='gray')
        # plt.show()

        # fig9 = plt.figure(9)
        # plt.imshow(hiddens_norm,cmap='gray')
        # plt.show()

        hiddens = hiddens.reshape(hiddens.shape[1],hiddens.shape[2],-1)

        # if (niter%50 == 0):
        #     for i in range(self.num_hiddens):
        #         fig = plt.figure()
        #         plt.imshow(hiddens[:,:,i],cmap='gray')
        #         plt.colorbar()
        #         #plt.savefig(path+'hiddens/hidden_img'+str(i))
        #         plt.close(fig)


        if (niter%10 ==0):
            fig1 = plt.figure(1)
            plt.imshow(entropy,vmin=0.0,vmax=5.0)
            plt.colorbar()
            # plt.show()
            plt.savefig(path+'random2/entropy_img'+str(niter))
            plt.close(fig1)


        return f_arr,gradient.reshape(*images.shape), whitened_images
		#return gradient.reshape(images.shape[1],images.shape[2])

    def gradient_mul(self, images, precond = None,niter=0,path=None,ent_max = 4.0):
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
                
        if (niter%10 ==0):
            for k in range(images.shape[0]):
                fig1 = plt.figure(1)
                plt.imshow(entropy[k,:,:].squeeze(),vmin=0.0,vmax=5.0)
                plt.colorbar()
                plt.savefig(path+str(k)+'/entropy_img'+str(niter))
                plt.close(fig1)

        return f_arr,gradient.reshape(*images.shape), whitened_images.reshape(*images.shape)

    def gradient_spc(self, images, precond = None,niter=0,path=None):
        """
        Returns the average log-likelihood [nat] and its gradient with respect to pixel values.

        @type  images: C{ndarray}
        @param images: images at which to evaluate the density's gradient

        @rtype: C{tuple}
        @return: average log-likelihood and gradient with respect to images
        """
        #Check

        sigma_s =15
        self.verbosity =0
        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)

        #if self.preconditioner:
        if self.verbosity > 0:
       
            print 'Preconditioning...'

        #self.preconditioner = None
        inputs_1 = copy(inputs)
        outputs_1 = copy(outputs)
        # remove correlations

        inputs_1, outputs_1 = self._precondition(inputs_1, outputs_1)
        if precond is not None :
            if self.preconditioner.pre_out > precond.pre_out:
                #print 'In condition',self.preconditioner.pre_out
                print 'Increased'
                #self.preconditioner = precond          
                #print 'trained pre_out',self.preconditioner.pre_out

        #self.preconditioner = None
        if self.preconditioner:
            inputs, outputs = self._precondition(inputs, outputs)

        # inputs = np.zeros_like(inputs) #CHANGE
        # outputs = np.zeros_like(outputs)

        if self.verbosity > 0:
            print 'Creating SLSTMs...'

        # create SLSTMs
        batch_size = min([images.shape[0], self.MAX_BATCH_SIZE])
        print self.slstm[0]
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


        # compute posterior
        posterior = self.mcgsm.posterior(H_flat, Y_flat)                
        #print posterior.sum(0).sum(), posterior.min(), posterior.max()
        log_posterior = np.log2(posterior)
        entropy = posterior*log_posterior
        
        entropy[np.isnan(entropy)] = 0
        entropy = entropy.sum(axis=0)        
        entropy = -entropy.T.reshape(outputs.shape[1],outputs.shape[2])
        # print 'sum', sum(sum(np.isnan(entropy)))
        # print 'entropy', entropy.min(), entropy.max()
        # print entropy.shape


        _l = np.zeros((images.shape[1],images.shape[2])) 
        _l[2:,2:-2] = np.copy(entropy)
        entropy = _l    
        
        # compute gradients
        df_dh, df_dy, loglik = self.mcgsm._data_gradient(H_flat, Y_flat)
        #print 'intial df_dh shape',df_dh.shape, 'intial df_dy shape',df_dy.shape
        df_dh = df_dh.T.reshape(*hiddens.shape) / H_flat.shape[1]
        df_dy = df_dy.T.reshape(*outputs.shape) / H_flat.shape[1]
        # average log-likelihood


        f = loglik/ H_flat.shape[1]
        #print 'f',f.sum()
        f = f.reshape(1,outputs.shape[1],outputs.shape[2],1) #Checl
        for l in range(self.num_layers)[::-1]:
            df_dh = self.slstm[l].backward(df_dh, force_backward=True)['inputs']

        df_dh_w =copy(df_dh)
        df_dy_w = copy(df_dy)
        if self.preconditioner:
            df_dh, df_dy = self._adjust_gradient(df_dh, df_dy)

        # locate output pixel in output mask
        for i_off, j_off in zip(
                range(self.output_mask.shape[0]),
                range(self.output_mask.shape[1])):
            if any(self.output_mask[i_off, j_off]):
                break

        gradient = zeros_like(images)
        gradient_w = zeros_like(images)
        whitened_images = copy(images)
        max_grad = 10000*np.zeros_like(images)
        # make sure mask and gradient have compatible dimensionality
        if gradient.ndim == 4 and self.input_mask.ndim == 2:
            gradient = gradient[:, :, :, 0]
            gradient_w = gradient_w[:,:,:,0]

        #if self.input_mask.ndim !=2:
        #    self.input_mask[:,:,0] = self.input_mask[:,:,-1]

        count = np.zeros(gradient.shape)
        #img = images.reshape(images.shape[1],images.shape[2])
        f_arr = np.zeros((1,images.shape[1],images.shape[2],1))
        #print 'input_mask shape',self.input_mask
        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                #ipm = copy(self.input_mask)

                # patch2 = count[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                patch3 = whitened_images[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                
                #ipm[patch2[0,:,:]>=2] = False
                # patch2[:, self.input_mask] += 1
                # patch2[:, self.output_mask] += 1
                patch3[:, self.output_mask] = outputs[:,i,j,:]
                f_arr[:,i+i_off,j+j_off,:] = f[:,i,j,:]

                # if i in [3,5] and j in [3,5]:
                    # print i,j
                    # print 'Input before',patch[:,self.input_mask]
                    # print 'df_dh',df_dh[:,i,j]
                # dfdh = df_dh[:,i,j]
                #print dfdh
                #print ipm.reshape(1,-1)[:,:12]
                #patch[:, self.input_mask] += dfdh*ipm.reshape(1,-1)[:,:12]
                #print 'dfdh',df_dh[:,i,j].shape
                if 1:#entropy[i+i_off,j+j_off] < 10000 :
                    patch = gradient[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                    patch4= gradient_w[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]

                    patch[:, self.input_mask] += df_dh[:,i,j]
                    patch[:, self.output_mask] += df_dy[:,i,j]
                    
                    patch4[:, self.input_mask] += df_dh_w[:,i,j]
                    patch4[:, self.output_mask] += df_dy_w[:,i,j]
                
                #gradient_w[i+i_off,j+j_off,:] = df_dy_w[:,i,j]
                # if 1:
                #     print 'i','j',i,j
                #     print 'df_dh',df_dh[:,i,j]
                #     print 'df_dy',df_dy[:,i,j]
                #     print gradient[:,i,j]
                # #Bilateral Filtering
                # input_weights = np.exp(-np.abs(img[self.input_mask]-img[self.output_mask]*(12))**2)/(2*sigma_s**2)
                # #print 'input weights shape',input_weights.shape
                # patch[:,self.input_mask] += input_weights*df_dh[:,i,j] 
                # output_weight = np.exp(-np.abs(np.mean(img[self.input_mask])-img[self.output_mask])**2)/(2*sigma_s**2)
                # patch[:,self.output_mask] += output_weight*df_dy[:,i,j]
                

                # patches_inp = patch3[:,self.input_mask].flatten() 
                # dfdh = df_dh[:,i,j].flatten()
                # #print dfdh.shape
                # #print patch3[:,self.input_mask].shape
                # for k,l1 in enumerate(patches_inp):
                #     patches_inp[k] = l1 if abs(l1) > abs(dfdh[k]) else dfdh[k]
                # patch3[:,self.input_mask] = patches_inp.reshape(*patch3[:,self.input_mask].shape)   
                # patches_op = patch3[:,self.output_mask]
                # patch3[:,self.output_mask] = patches_op if np.abs(patches_op) > np.abs(df_dy[:,i,j]) else df_dy[:,i,j]

                # if i in [a3,5] and j in [3,5]:    
                    # print 'Input after',patch[:,self.input_mask]
        # count[count==0] = 1
        #gradient = np.multiply(gradient,1.0/count.astype(float))
        #print 'gradient shape',gradient.shape
        #print 'Hidden grad', np.abs(gradient_w).sum()
        #print inputs.shape

        # for l in range(self.num_layers):
        #     print'h_init_i_jm1', self.slstm[l].net.blobs['h_init_i_jm1'].data[:] 
        #     print 'h_init_im1_j',self.slstm[l].net.blobs['h_init_im1_j'].data[:]
        #     print 'c_init_i_jm1',self.slstm[l].net.blobs['c_init_i_jm1'].data[:]
        #     print 'c_init_im1_j',self.slstm[l].net.blobs['c_init_im1_j'].data[:]


        inputs_norm = np.linalg.norm(inputs.reshape(inputs.shape[1],inputs.shape[2],-1),axis = 2)
        #grad_h_norm = np.linalg.norm(gradient_w,axis=2)
        hiddens_1_norm = np.linalg.norm(hiddens_1.reshape(hiddens_1.shape[1],hiddens_1.shape[2],-1),axis = 2)
        hiddens_norm = np.linalg.norm(hiddens.reshape(hiddens.shape[1],hiddens.shape[2],-1),axis = 2)
        
        # fig5 = plt.figure(5)
        # xx,yy = np.mgrid[0:hiddens_norm.shape[0],0:hiddens_norm.shape[1]]
        # ax = fig5.gca(projection='3d')
        # ax.plot_surface(xx,yy,inputs_norm,cmap=cm.coolwarm,linewidth=0)
        # plt.show()

        # fig6 = plt.figure(6)
        # xx,yy = np.mgrid[0:hiddens_1_norm.shape[0],0:hiddens_1_norm.shape[1]]
        # ax = fig6.gca(projection='3d')
        # ax.plot_surface(xx,yy,hiddens_1_norm,cmap=cm.coolwarm,linewidth=0)
        # plt.show()

        # fig7 = plt.figure(7)
        # xx,yy = np.mgrid[0:hiddens_norm.shape[0],0:hiddens_norm.shape[1]]
        # ax = fig7.gca(projection='3d')
        # ax.plot_surface(xx,yy,hiddens_norm,cmap=cm.coolwarm,linewidth=0)
        # plt.show()

        # fig8 = plt.figure(8)
        # plt.imshow(hiddens_norm,cmap='gray')
        # plt.show()

        # fig9 = plt.figure(9)
        # plt.imshow(hiddens_norm,cmap='gray')
        # plt.show()

        hiddens = hiddens.reshape(hiddens.shape[1],hiddens.shape[2],-1)

        # if (niter%50 == 0):
        #     for i in range(self.num_hiddens):
        #         fig = plt.figure()
        #         plt.imshow(hiddens[:,:,i],cmap='gray')
        #         plt.colorbar()
        #         #plt.savefig(path+'hiddens/hidden_img'+str(i))
        #         plt.close(fig)


        if (niter%10 ==0):
            fig1 = plt.figure(1)
            plt.imshow(entropy,vmin=0.0,vmax=5.0)
            plt.colorbar()
            # plt.show()
            plt.savefig(path+'random1/entropy_img'+str(niter))
            plt.close(fig1)



        return f_arr,gradient.reshape(*images.shape), whitened_images,gradient_w.reshape(*images.shape),entropy
        #return gradient.reshape(images.shape[1],images.shape[2])

    def loss_gradient(self, images,noisy_images, precond = None,niter=0):
        """
        Returns the average log-likelihood [nat] and its gradient with respect to pixel values.

        @type  images: C{ndarray}
        @param images: images at which to evaluate the density's gradient

        @rtype: C{tuple}
        @return: average log-likelihood and gradient with respect to images
        """
        #Check

        sigma_s =15
        self.verbosity =0
        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)
        noisy_inputs,noisy_outputs = self._preprocess(noisy_images)

        #if self.preconditioner:
        if self.verbosity > 0:
       
            print 'Preconditioning...'

        # #self.preconditioner = None
        # inputs_1 = copy(inputs)
        # outputs_1 = copy(outputs)
        # # remove correlations

        # inputs_1, outputs_1 = self._precondition(inputs_1, outputs_1)
        # if precond is not None :
        #     if self.preconditioner.pre_out > precond.pre_out:
        #         #print 'In condition',self.preconditioner.pre_out
        #         print 'Increased'
        #         #self.preconditioner = precond          
        #         #print 'trained pre_out',self.preconditioner.pre_out

        #self.preconditioner = None
        if self.preconditioner:
            inputs, outputs = self._precondition(inputs, outputs)
            noisy_inputs,noisy_outputs = self._precondition(noisy_inputs,noisy_outputs)

        # inputs = np.zeros_like(inputs) #CHANGE
        # outputs = np.zeros_like(outputs)

        if self.verbosity > 0:
            print 'Creating SLSTMs...'

        # create SLSTMs
        batch_size = min([images.shape[0], self.MAX_BATCH_SIZE])
        print self.slstm[0]
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

        # compute hidden unit activations
        noisy_hiddens = noisy_inputs
        for l in range(self.num_layers):
            noisy_hiddens = self.slstm[l].forward(noisy_hiddens)
            if l == 0:
                noisy_hiddens_1 = noisy_hiddens


        # form inputs to MCGSM
        H_flat = hiddens.reshape(-1, self.num_hiddens).T
        Y_flat = outputs.reshape(-1, self.num_channels).T
        #print 'outputs shape',outputs.shape

        # # compute gradients
        df_dh, df_dy, loglik = self.mcgsm._data_gradient(H_flat, Y_flat)
        #print 'intial df_dh shape',df_dh.shape, 'intial df_dy shape',df_dy.shape
        df_dh = df_dh.T.reshape(*hiddens.shape) / H_flat.shape[1]
        df_dy = df_dy.T.reshape(*outputs.shape) / H_flat.shape[1]


        df_dh = 2*(noisy_hiddens - hiddens)/ (H_flat.shape[1]) 
        df_dy = np.zeros((1,hiddens.shape[1],hiddens.shape[2],1))   

        # print 'df_dh shape', df_dh.shape
        # print 'df_dy shape',df_dy.shape
        # print 'hiddens shape',hiddens.shape


        f = np.sum(-1*(hiddens-noisy_hiddens)**2/ H_flat.shape[1],axis = -1)
        f = f.reshape(1,outputs.shape[1],outputs.shape[2],1) 

        #Checl
        for l in range(self.num_layers)[::-1]:
            df_dh = self.slstm[l].backward(df_dh, force_backward=True)['inputs']


        if self.preconditioner:
            df_dh, df_dy = self._adjust_gradient(df_dh, df_dy)

        # locate output pixel in output mask
        for i_off, j_off in zip(
                range(self.output_mask.shape[0]),
                range(self.output_mask.shape[1])):
            if any(self.output_mask[i_off, j_off]):
                break

        gradient = zeros_like(images)
        gradient_w = zeros((images.shape[1],images.shape[2],self.num_hiddens))
        whitened_images = copy(images)
        max_grad = 10000*np.zeros_like(images)
        # make sure mask and gradient have compatible dimensionality
        if gradient.ndim == 4 and self.input_mask.ndim == 2:
            gradient = gradient[:, :, :, 0]
            #gradient_w = gradient_w[:,:,:,0]

        #if self.input_mask.ndim !=2:
        #    self.input_mask[:,:,0] = self.input_mask[:,:,-1]

        count = np.zeros(gradient.shape)
        #img = images.reshape(images.shape[1],images.shape[2])
        f_arr = np.zeros((1,images.shape[1],images.shape[2],1))
        print df_dh_w.shape
        #print 'input_mask shape',self.input_mask
        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                #ipm = copy(self.input_mask)
                patch = gradient[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                patch2 = count[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                patch3 = whitened_images[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]

                #ipm[patch2[0,:,:]>=2] = False
                patch2[:, self.input_mask] += 1
                patch2[:, self.output_mask] += 1
                patch3[:, self.output_mask] = outputs[:,i,j,:]
                f_arr[:,i+i_off,j+j_off,:] = f[:,i,j,:]

                # if i in [3,5] and j in [3,5]:
                    # print i,j
                    # print 'Input before',patch[:,self.input_mask]
                    # print 'df_dh',df_dh[:,i,j]
                # dfdh = df_dh[:,i,j]
                #print dfdh
                #print ipm.reshape(1,-1)[:,:12]
                #patch[:, self.input_mask] += dfdh*ipm.reshape(1,-1)[:,:12]
                #print 'dfdh',df_dh[:,i,j].shape
                patch[:, self.input_mask] += df_dh[:,i,j]
                patch[:, self.output_mask] += df_dy[:,i,j]
                
                gradient_w[i+i_off,j+j_off,:] = df_dy_w[:,i,j]
                # if 1:
                #     print 'i','j',i,j
                #     print 'df_dh',df_dh[:,i,j]
                #     print 'df_dy',df_dy[:,i,j]
                #     print gradient[:,i,j]
                # #Bilateral Filtering
                # input_weights = np.exp(-np.abs(img[self.input_mask]-img[self.output_mask]*(12))**2)/(2*sigma_s**2)
                # #print 'input weights shape',input_weights.shape
                # patch[:,self.input_mask] += input_weights*df_dh[:,i,j] 
                # output_weight = np.exp(-np.abs(np.mean(img[self.input_mask])-img[self.output_mask])**2)/(2*sigma_s**2)
                # patch[:,self.output_mask] += output_weight*df_dy[:,i,j]
                

                # patches_inp = patch3[:,self.input_mask].flatten() 
                # dfdh = df_dh[:,i,j].flatten()
                # #print dfdh.shape
                # #print patch3[:,self.input_mask].shape
                # for k,l1 in enumerate(patches_inp):
                #     patches_inp[k] = l1 if abs(l1) > abs(dfdh[k]) else dfdh[k]
                # patch3[:,self.input_mask] = patches_inp.reshape(*patch3[:,self.input_mask].shape)   
                # patches_op = patch3[:,self.output_mask]
                # patch3[:,self.output_mask] = patches_op if np.abs(patches_op) > np.abs(df_dy[:,i,j]) else df_dy[:,i,j]

                # if i in [a3,5] and j in [3,5]:    
                    # print 'Input after',patch[:,self.input_mask]
        count[count==0] = 1
        #gradient = np.multiply(gradient,1.0/count.astype(float))
        #print 'gradient shape',gradient.shape
        print 'Hidden grad', np.abs(gradient_w).sum()
        #print inputs.shape

        # for l in range(self.num_layers):
        #     print'h_init_i_jm1', self.slstm[l].net.blobs['h_init_i_jm1'].data[:] 
        #     print 'h_init_im1_j',self.slstm[l].net.blobs['h_init_im1_j'].data[:]
        #     print 'c_init_i_jm1',self.slstm[l].net.blobs['c_init_i_jm1'].data[:]
        #     print 'c_init_im1_j',self.slstm[l].net.blobs['c_init_im1_j'].data[:]


        inputs_norm = np.linalg.norm(inputs.reshape(inputs.shape[1],inputs.shape[2],-1),axis = 2)
        grad_h_norm = np.linalg.norm(gradient_w,axis=2)
        hiddens_1_norm = np.linalg.norm(hiddens_1.reshape(hiddens_1.shape[1],hiddens_1.shape[2],-1),axis = 2)
        hiddens_norm = np.linalg.norm(hiddens.reshape(hiddens.shape[1],hiddens.shape[2],-1),axis = 2)
        
        # fig5 = plt.figure(5)
        # xx,yy = np.mgrid[0:hiddens_norm.shape[0],0:hiddens_norm.shape[1]]
        # ax = fig5.gca(projection='3d')
        # ax.plot_surface(xx,yy,inputs_norm,cmap=cm.coolwarm,linewidth=0)
        # plt.show()

        # fig6 = plt.figure(6)
        # xx,yy = np.mgrid[0:hiddens_1_norm.shape[0],0:hiddens_1_norm.shape[1]]
        # ax = fig6.gca(projection='3d')
        # ax.plot_surface(xx,yy,hiddens_1_norm,cmap=cm.coolwarm,linewidth=0)
        # plt.show()

        # fig7 = plt.figure(7)
        # xx,yy = np.mgrid[0:hiddens_norm.shape[0],0:hiddens_norm.shape[1]]
        # ax = fig7.gca(projection='3d')
        # ax.plot_surface(xx,yy,hiddens_norm,cmap=cm.coolwarm,linewidth=0)
        # plt.show()

        # fig8 = plt.figure(8)
        # plt.imshow(hiddens_norm,cmap='gray')
        # plt.show()



        return f_arr,gradient.reshape(*images.shape), whitened_images
        #return gradient.reshape(images.shape[1],images.shape[2])


    def gradient_1(self, images):
        """
        Returns the average log-likelihood [nat] and its gradient with respect to pixel values.

        @type  images: C{ndarray}
        @param images: images at which to evaluate the density's gradient

        @rtype: C{tuple}
        @return: average log-likelihood and gradient with respect to images
        """
        images1= copy(images)
        images = images.reshape(1,np.sqrt(images.shape[0]),np.sqrt(images.shape[0]),1)
        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)

        if self.preconditioner:
            if self.verbosity > 0:
                print 'Preconditioning...'

            # remove correlations
            inputs, outputs = self._precondition(inputs, outputs)

        if self.verbosity > 0:
            print 'Creating SLSTMs...'

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

        # form inputs to MCGSM
        H_flat = hiddens.reshape(-1, self.num_hiddens).T
        Y_flat = outputs.reshape(-1, self.num_channels).T

        # compute gradients
        df_dh, df_dy, loglik = self.mcgsm._data_gradient(H_flat, Y_flat)
        #print 'intial df_dh shape',df_dh.shape, 'intial df_dy shape',df_dy.shape
        df_dh = df_dh.T.reshape(*hiddens.shape) / H_flat.shape[1]
        df_dy = df_dy.T.reshape(*outputs.shape) / H_flat.shape[1]

          
        f = loglik / H_flat.shape[1]
         

        for l in range(self.num_layers)[::-1]:
            df_dh = self.slstm[l].backward(df_dh, force_backward=True)['inputs']

        if self.preconditioner:
            df_dh, df_dy = self._adjust_gradient(df_dh, df_dy)

        # locate output pixel in output mask
        for i_off, j_off in zip(
                range(self.output_mask.shape[0]),
                range(self.output_mask.shape[1])):
            if any(self.output_mask[i_off, j_off]):
                break

        gradient = zeros_like(images)

        # make sure mask and gradient have compatible dimensionality
        if gradient.ndim == 4 and self.input_mask.ndim == 2:
            gradient = gradient[:, :, :, 0]

        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                patch = gradient[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                patch[:, self.input_mask] += df_dh[:,i,j]
                patch[:, self.output_mask] += df_dy[:,i,j]

        return f,gradient.reshape(*images1.shape)
		#return gradient.reshape(images.shape[1],images.shape[2])

    def gradient_rgb(self, images):
        """
        Returns the average log-likelihood [nat] and its gradient with respect to pixel values.

        @type  images: C{ndarray}
        @param images: images at which to evaluate the density's gradient

        @rtype: C{tuple}
        @return: average log-likelihood and gradient with respect to images
        """
        #mplimg.imsave('/home/cplab-ws1/ride/code/map_interpolate/'+'images_before_gradientrgb.png',images)
        #if self.verbosity > 0:
        #    print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)
        self.verbosity = 0
        if self.preconditioner:
            if self.verbosity > 0:
                print 'Preconditioning...'
            # remove correlations
            inputs, outputs = self._precondition(inputs, outputs)

        #if self.verbosity > 0:
        #    print 'Creating SLSTMs...'

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

        # form inputs to MCGSM
        H_flat = hiddens.reshape(-1, self.num_hiddens).T
        Y_flat = outputs.reshape(-1, self.num_channels).T

        # compute gradients
        df_dh, df_dy, loglik = self.mcgsm._data_gradient(H_flat, Y_flat)
        #print 'intial df_dh shape',df_dh.shape, 'intial df_dy shape',df_dy.shape
        df_dh = df_dh.T.reshape(*hiddens.shape) / H_flat.shape[1]
        df_dy = df_dy.T.reshape(*outputs.shape) / H_flat.shape[1]

        #print 'df_dh shape',df_dh.shape
        #print df_dh.sum()
        #df_dh = np.zeros_like(df_dh)

        f = sum(loglik) / H_flat.shape[1]

        for l in range(self.num_layers)[::-1]:
            df_dh = self.slstm[l].backward(df_dh, force_backward=True)['inputs']
            #print 'df_dh ',df_dh.sum()  
        if self.preconditioner:
            #print 'Adjusting gradients'
            df_dh, df_dy = self._adjust_gradient(df_dh, df_dy)

        # locate output pixel in output mask
        for i_off, j_off in zip(
                range(self.output_mask.shape[0]),
                range(self.output_mask.shape[1])):
            if any(self.output_mask[i_off, j_off]):
                break

        gradient = zeros_like(images)
        
        #print 'gradient shape',gradient.shape
        #print 'input_mask shape',self.input_mask.shape

        # make sure mask and gradient have compatible dimensionality
        if gradient.ndim == 4 and self.input_mask.ndim == 2:
            #print 'Compatible dimensions'
            gradient = gradient[:, :, :, 0]

        #count = np.zeros(gradient.shape)
        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                #ipm = copy(self.input_mask)
                patch = gradient[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                #patch2 = count[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                #patch3 = max_grad[:, i:i + self.input_mask.shape[0], j:j + self.output_mask.shape[1]]
                #ipm[patch2[0,:,:]>=2] = False
                #patch2[:, self.input_mask] += 1
                #patch2[:, self.output_mask] += 1
                # if i in [3,5] and j in [3,5]:
                    # print i,j
                    # print 'Input before',patch[:,self.input_mask]
                    # print 'df_dh',df_dh[:,i,j]
                # dfdh = df_dh[:,i,j]
                #print dfdh
                #print ipm.reshape(1,-1)[:,:12]
                #patch[:, self.input_mask] += dfdh*ipm.reshape(1,-1)[:,:12]
                #print 'Portion of patch to be added' ,patch[:,self.input_mask].shape
                #print df_dh[:,i,j].shape
                #print patch[:,self.input_mask].shape
                patch[:, self.input_mask] += df_dh[:,i,j]
                patch[:, self.output_mask] += df_dy[:, i, j]

        #count[count==0] = 1
        #gradient = np.multiply(gradient,1.0/count.astype(float))
        #print self.output_mask.sum()
        #print 'loglik', f
        #mplimg.imsave('/home/cplab-ws1/ride/code/map_interpolate/'+'images_after_gradientrgb.png',images)
        return f,gradient.reshape(*images.shape)

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

    def map_denoise(self, noisy_img, path=None, noise_std=0.1, lr=0.01, niter=100):
        """
        Returns the average log-likelihood [nat] and its gradient with respect to pixel values.

        @type  images: C{ndarray}
        @param images: images at which to evaluate the density's gradient

        @rtype: C{tuple}
        @return: average log-likelihood and gradient with respect to images
        """
        import matplotlib.image as mplimg
        from matplotlib import cm

        from scipy import ndimage
        blur_img = ndimage.gaussian_filter(noisy_img, sigma=3)
        images = blur_img
        
        if self.verbosity > 0:
            print 'Preprocessing...'

        inputs, outputs = self._preprocess(images)
        noisy_inputs, noisy_outputs = self._preprocess(noisy_img) 
        #print 'inputs', inputs.shape
        if self.preconditioner:
            if self.verbosity > 0:
                print 'Preconditioning...'

            # remove correlations
            inputs, outputs = self._precondition(inputs, outputs)
            noisy_inputs, noisy_outputs = self._precondition(noisy_inputs, noisy_outputs)

        
        pre_in = self.preconditioner.pre_in
        pre_out = self.preconditioner.pre_out
        predictor = self.preconditioner.predictor
        
        b = pre_out * np.dot(predictor,pre_in)
        b = np.square(b)
        whitened_noise_std = noise_std * np.sqrt(pre_out**2 + b.sum())
        print 'whitened_noise_std', whitened_noise_std

        if self.verbosity > 0:
            print 'Creating SLSTMs...'

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

        _img = self.get_image(inputs,outputs,images.shape, grad=False)
        #print 'shape', inputs.shape, outputs.shape

        for i in range(niter):
            if(i%10==0):
                #lr = 0.99*lr
                #print 'lr', lr
                t1 = _img
                _img = self.get_image(inputs,outputs,images.shape, grad=False)
                print _img.shape, np.abs(t1-_img).sum()
                mplimg.imsave(path+'cleaned_img'+str(i), _img.reshape(images.shape[1],images.shape[2]), cmap = cm.gray)
            print i, 

            # compute hidden unit activations
            hiddens = np.copy(inputs)
            #print 'forwarding the data'
            for l in range(self.num_layers):
                hiddens = self.slstm[l].forward(hiddens)

            # form inputs to MCGSM
            H_flat = hiddens.reshape(-1, self.num_hiddens).T
            Y_flat = outputs.reshape(-1, self.num_channels).T

            # compute gradients
            df_dh, df_dy, loglik = self.mcgsm._data_gradient(H_flat, Y_flat)
            df_dh = df_dh.T.reshape(*hiddens.shape) / H_flat.shape[1]
            df_dy = df_dy.T.reshape(*outputs.shape) / H_flat.shape[1]

            # average log-likelihood
            f = sum(loglik) / H_flat.shape[1]
            for l in range(self.num_layers)[::-1]:
                df_dh = self.slstm[l].backward(df_dh, force_backward=True)['inputs']
    
            print 'Loglikelihood', f, 'input_grad', np.abs(df_dh).sum(), 
            print 'output_grad', np.abs(df_dy).sum()
            #tmp_in = 
            #print np.abs(tmp_in).sum(), 
            #tmp_in1 = lr*(noisy_inputs - inputs)/(whitened_noise_std**2)
            #print np.abs(tmp_in1).sum()
            #print 'df_dh shape',df_dh.shape
            #print 'df_dy shape',df_dy.shape 

            inputs = inputs + lr*df_dh #+ lr*(noisy_inputs - inputs)/(whitened_noise_std**2)
            outputs = outputs + lr*df_dy #+ lr*(noisy_outputs - outputs)/(whitened_noise_std**2)
            #print outputs.shape

            #img = np.copy(noisy_img)
            #print img.shape
            #img[0,self.input_mask.shape[0]-1:blur_img.shape[1], self.input_mask.shape[0]-1: blur_img.shape[2] - self.input_mask.shape[0]+1,0] = \
            #outputs.reshape(outputs.shape[1],outputs.shape[2])

            #inputs, outputs = self._preprocess(img)
            #t_img = self.get_image(inputs, outputs, images.shape, grad=True)
            #inputs, outputs = self._preprocess(t_img)
            #print inputs.shape, outputs.shape
                    
        gradient = self.get_image(inputs, outputs, images.shape, grad=False)
        return gradient.reshape(*images.shape)


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


        
    def interpolate(self,
            images,
            binary_mask,
            min_values=None,
            max_values=None,
            mask=None,
            return_loglik=False,
            method = 'mode',
            sample = 10,
            lr=0.01):
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
                
        sampled_images = deepcopy(images)

        # create spatial LSTMs for sampling
        # print 'creating lstm'
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

        avg_logl = []
        #p_exp_arr = []
        
        # container for hidden and memory unit activations
        hiddens = []
        memory = []
        hid_arr = []
        mem_arr = [] 
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

        h_norm =  zeros([images.shape[1] - self.input_mask.shape[0] + 1,images.shape[2] - self.input_mask.shape[1] + 1])
        c_norm =  zeros([images.shape[1] - self.input_mask.shape[0] + 1,images.shape[2] - self.input_mask.shape[1] + 1])
        tanh_c =  zeros([images.shape[1] - self.input_mask.shape[0] + 1,images.shape[2] - self.input_mask.shape[1] + 1])

        if method == 'mean' or method == 'max' :
            #distribution params
            priors = np.array(self.mcgsm.priors)
            scales = self.mcgsm.scales
            weights = self.mcgsm.weights
            features = self.mcgsm.features
            cholesky_factors = self.mcgsm.cholesky_factors
            predictors = np.array(self.mcgsm.predictors)
            predictors = predictors.reshape(self.mcgsm.num_components,self.num_channels,-1)
            expert_std = np.sqrt(1.0/np.exp(scales))
                       
        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            hid_arr.append([])
            mem_arr.append([])
            avg_logl.append([])
            # print 'i',i
            #p_exp_arr.append([])
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                #print 'j',j
                # extract patches from images
                # print 'patch extraction'
                patches = images[:,
                    i:i + self.input_mask.shape[0],
                    j:j + self.input_mask.shape[1]]

                # patches_mask = np.array([binary_mask[
                #     i:i + self.input_mask.shape[0],
                #     j:j + self.input_mask.shape[1]]])

                # extract causal neighborhoods from patches
                inputs = []
                outputs = []
                # input_masks =[]
                for k in range(images.shape[0]):
                    patches_inputs,patches_outputs = generate_data_from_image(
                            patches[k, :, :], self.input_mask, self.output_mask)
                    inputs.append(patches_inputs)
                    outputs.append(patches_outputs)
                    # input_masks.append(generate_data_from_image(
                            # patches_mask[k, :, :], self.input_mask, self.output_mask)[0])
                inputs = asarray(inputs)
                outputs = asarray(outputs)
                # input_masks = asarray(input_masks)

                inputs = inputs.reshape(inputs.shape[0], 1, 1, -1)
                outputs = outputs.reshape(outputs.shape[0],1,1,-1)
                # input_masks = input_masks.reshape(1,1,1,-1)
                
                if self.preconditioner:
                    inputs,outputs = self._precondition(inputs,outputs)

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
                hid_arr[i].append(hiddens[-1][i,j])
                mem_arr[i].append(memory[-1][i,j])

                #outputs = None
                if method == 'mode':
                    max_logq = -10
                    avg_f = np.zeros(sample)
                    for it in range(sample):
                        # sample MCGSM
                        # print 'Hiddens shape',hiddens[-1][i, j].reshape(-1, self.num_hiddens).T.shape
    
                        # print 'Inputs shape',inputs.reshape(-1, 12).T.shape

                        outputs = self.mcgsm.sample(
                            hiddens[-1][i, j].reshape(-1, self.num_hiddens).T)
                                            
                        logq = self.mcgsm.loglikelihood(
                            hiddens[-1][i, j].reshape(-1, self.num_hiddens).T,
                            outputs)
                        avg_f[it] = logq
                        
                        if logq > max_logq:
                            max_logq = logq
                            #print logq
                            ml_outputs = outputs
                    
                        outputs = ml_outputs                        
                    avg_logl[i].append(avg_f)

                elif method =='max' :
                    #l_r = 0.01     #0.01,20  0.0001,30
                    #sample = 30
                    
                    avg_f = np.zeros(sample)
                    f = -100
                    for it in range(sample):
                        
                        if it > 0:                        
                            ########Forward_pass                                    
                            activations = inputs
                            for l in range(self.num_layers):
                                activations = self.slstm[l].forward(activations)
                                                # store hidden unit activations
                            for l in range(self.num_layers):
                                hiddens[l][i, j] = self.slstm[l].net.blobs['outputs'].data.copy()
                                memory[l][i, j] = self.slstm[l].net.blobs['c_0_0'].data.copy()
                            
                        H_flat = hiddens[-1][i, j].reshape(-1, self.num_hiddens).T                         
                                              
                        if 0:#not binary_mask[i+i_off,j+j_off]: #unknown point
                            ###########Expected value
                            # component gate probabilities
                            square_weights = np.square(weights)
                            feature_input = np.dot(features.T,H_flat)
                            gate_energy = np.dot(square_weights,np.square(feature_input))
                            gate_energy = priors - 0.5* np.multiply(np.exp(scales), gate_energy)
                            prior_prob = np.exp(gate_energy)
                            prior_prob = prior_prob/prior_prob.sum()
                            
                            #Finding out the expected value                    
                            #For 1 dimensional output
                            #means = np.tile(np.dot(predictors,H_flat),scales.shape[1])
                            #outputs = np.asarray([[(means*prior_prob).sum()]])
                            #For n dimensional output
                            outputs = np.dot(prior_prob.transpose(),np.dot(predictors,H_flat)).sum(axis = 0)
                            #Check here for exact shape                        
                            Y_flat = outputs.reshape(-1,self.num_channels).T
                            break
                            
                        else:
                            Y_flat = outputs.reshape(-1,self.num_channels).T
                        
                        #Computing expert prob for loglik
                        #expert_means = np.dot(predictors,H_flat)
                        #for k in range(self.mcgsm.num_scales-1):
                        #    expert_means = np.append(expert_means,np.dot(predictors,H_flat), axis = 1)

                        #prob_obs = np.zeros((self.mcgsm.num_components, self.mcgsm.num_scales))
                        #for r in range(self.mcgsm.num_components):
                        #    for c in range(self.mcgsm.num_scales):
                        #        prob_obs[r,c] = scipy.stats.norm(expert_means[r,c], expert_std[r,c]).pdf(Y_flat.reshape(1))
                        
                        #loglikim = np.log((prior_prob*prob_obs).sum())
                        #print loglikim
                        
                        
                        ##########Backward pass
                        # form inputs to MCGSM
                        #compute gradeints
                        df_dh, df_dy, loglik = self.mcgsm._data_gradient(H_flat, Y_flat)
                        df_dh = np.array([df_dh.T.reshape(*hiddens[-1][i,j].shape)]) / H_flat.shape[1]
                        df_dy = df_dy.T.reshape(*outputs.shape) / H_flat.shape[1]
                                                
    
                        # average log-likelihood
                        #f_prev = f
                        f = sum(loglik) / H_flat.shape[1]
                        avg_f[it] = f
                        #print it,'loglik',f
                        
                        #if f_prev > f:
                        #    break
    
                        for l in range(self.num_layers)[::-1]:
                            df_dh = self.slstm[l].backward(df_dh, force_backward=True)['inputs']
                        if it > 0:
                            inputs += (1-input_masks)*(lr*df_dh) 

                    avg_logl[i].append(avg_f)
                    #p_exp_arr[i].append(loglikim)

                if method == 'mean' :
                    whitened_nbd = hiddens[-1][i, j].reshape(-1, self.num_hiddens).T    
                    
                    # component gate probabilities
                    square_weights = np.square(weights)
                    feature_input = np.dot(features.T,whitened_nbd)
                    gate_energy = np.dot(square_weights,np.square(feature_input))
                    gate_energy = priors - 0.5* np.multiply(np.exp(scales), gate_energy)
                    prior_prob = np.exp(gate_energy)
                    prior_prob = prior_prob/prior_prob.sum()
                    
                    #Finding out the expected value                    
                    #one dimensional output
                    # means = np.tile(np.dot(predictors,whitened_nbd),scales.shape[1])
                    # outputs = np.asarray([[(means*prior_prob).sum()]]) #Check here for exact shape
                    #n dimensional output
                    outputs = np.dot(prior_prob.transpose(),np.dot(predictors,whitened_nbd)[:,:,0]).sum(axis = 0)

                    logq = self.mcgsm.loglikelihood(
                            whitened_nbd,outputs)
                    #print 'logq',logq
                    avg_logl[i].append(np.asarray([logq]))
                    
                outputs = outputs.T.reshape(1,1,1,self.num_channels)    
                if self.preconditioner:
                    inputs, outputs = self._precondition_inverse(inputs, outputs)
                #print 'inputs max',inputs.max()
                if method =='max1':
                    for i1 in range(i,i+self.input_mask.shape[0]):
                        for j1 in range(j,j+self.input_mask.shape[1]):
                            if (self.input_mask[i1-i,j1-j]):                            
                               images[:,i1,j1] = inputs[:,:,:,(i1-i)*(self.input_mask.shape[1])+j1-j]   #Check shape                 
                
                #If it already has value
                # print i_off,j_off,i,j
                # print binary_mask.shape
                if binary_mask[i+ i_off,j+j_off].any():                
                    continue  
                    
                #print outputs
			
                if max_values is not None:
                    outputs[outputs > max_values] = max_values[outputs > max_values]
                if min_values is not None:
                    outputs[outputs < min_values] = min_values[outputs < min_values]                
                # insert sampled pixels into images
                #if self.output_mask.ndim > 2:
                #    images[:, i + i_off, j + j_off][:, self.output_mask[i_off, j_off]] = outputs
                #else:
                #images[:, i + i_off, j + j_off] = outputs
                
                if self.output_mask.ndim > 2:
                    images[:, i + i_off, j + j_off][:, self.output_mask[i_off, j_off]] = outputs
                else:
                    images[:, i + i_off, j + j_off] = outputs        

        images = images.reshape(*shape)
        sampled_images = sampled_images.reshape(*shape)
        
        #np.save('p_exp.npy',p_exp_arr)
        np.save('hiddens.npy',hid_arr)
        np.save('memory.npy',mem_arr)
        np.save('avg_logl.npy',avg_logl)
        sample_act = {'h':h_norm,'c':c_norm,'t':tanh_c}
        savemat('sample_act.mat',sample_act)

        return images,sampled_images,np.array(avg_logl)[:,:,-1]


    def denoise(self,
            images,
            min_values=None,
            max_values=None,
            mask=None,
            return_loglik=False,
            method = 'max',
            sample = 10,
            lr=0.1,
            sigma=0.08,
            clean_images = None):
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
        
        if clean_images.ndim == 2:
            clean_images = clean_images[None, :, :, None]
        elif clean_images.ndim == 3:
            if self.num_channels > 1:
                clean_images = clean_images[None]
            else:
                clean_images = clean_images[:, :, :, None]

        sampled_images = copy(images)

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

        avg_logl = []
        # container for hidden and memory unit activations
        hiddens = []
        memory = []
        #hid_arr = []
        #mem_arr = [] 
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

        h_norm =  zeros([images.shape[1] - self.input_mask.shape[0] + 1,images.shape[2] - self.input_mask.shape[1] + 1])
        c_norm =  zeros([images.shape[1] - self.input_mask.shape[0] + 1,images.shape[2] - self.input_mask.shape[1] + 1])
        tanh_c =  zeros([images.shape[1] - self.input_mask.shape[0] + 1,images.shape[2] - self.input_mask.shape[1] + 1])

        #distribution params
        priors = np.array(self.mcgsm.priors)
        scales = self.mcgsm.scales
        weights = self.mcgsm.weights
        features = self.mcgsm.features
        cholesky_factors = self.mcgsm.cholesky_factors
        predictors = np.array(self.mcgsm.predictors)
        predictors = predictors.reshape(self.mcgsm.num_components,self.num_channels,-1)
        expert_std = np.sqrt(1.0/np.exp(scales))
        
        pre_in = self.preconditioner.pre_in # size nbd x nbd
        #mean_out = self.preconditioner.mean_out # size  1,1
        #print mean_out
        pre_out = self.preconditioner.pre_out # size 1,1
        #print pre_out
        predictor = self.preconditioner.predictor            
        
        b =np.dot(pre_out,(np.dot(predictor, pre_in)))
       
        #For 1 dimensional output
        if pre_out.shape[0] == 1:
            b = np.square(b)
            whitened_noise_std = sigma * np.sqrt(pre_out**2 + b.sum())
            print 'whitened_sigma',whitened_noise_std
        #For n dimensional output
        else:
            wht_cov = sigma**2*(np.dot(pre_out,pre_out.transpose())+np.dot(b,b.transpose()))

        for i in range(images.shape[1] - self.input_mask.shape[0] + 1):
            #hid_arr.append([])
            #mem_arr.append([])
            avg_logl.append([])
            print 'i',i
            #p_exp_arr.append([])
            for j in range(images.shape[2] - self.input_mask.shape[1] + 1):
                #print 'j',j
                # extract patches from images
                # print 'patch extraction'
                patches = images[:,
                    i:i + self.input_mask.shape[0],
                    j:j + self.input_mask.shape[1]]
                clean_patches = clean_images[:,
                    i:i + self.input_mask.shape[0],
                    j:j + self.input_mask.shape[1]]
                generated_patches = sampled_images[:,
                    i:i + self.input_mask.shape[0],
                    j:j + self.input_mask.shape[1]]

                # extract causal neighborhoods from patches
                inputs = []
                outputs = []
                clean_inputs= []
                clean_outputs = []
                generated_inputs= []
                generated_outputs = []

                
                for k in range(images.shape[0]):
                    patches_inputs,patches_outputs = generate_data_from_image(
                            patches[k, :, :], self.input_mask, self.output_mask)
                    clean_patches_inputs,clean_patches_outputs = generate_data_from_image(
                            clean_patches[k, :, :], self.input_mask, self.output_mask)
                    generated_patches_inputs,generated_patches_outputs = generate_data_from_image(
                            generated_patches[k, :, :], self.input_mask, self.output_mask)
                    
                    inputs.append(patches_inputs)
                    outputs.append(patches_outputs)
                    
                    clean_inputs.append(clean_patches_inputs)
                    clean_outputs.append(clean_patches_outputs)
                    
                    generated_inputs.append(generated_patches_inputs)
                    generated_outputs.append(generated_patches_outputs)

                inputs = asarray(inputs)
                outputs = asarray(outputs)

                clean_inputs = asarray(clean_inputs)    
                clean_outputs = asarray(clean_outputs)

                generated_inputs = asarray(generated_inputs)
                generated_outputs = asarray(generated_outputs)

                inputs = inputs.reshape(inputs.shape[0], 1, 1, -1)
                outputs = outputs.reshape(outputs.shape[0],1,1,-1)
                
                clean_inputs = clean_inputs.reshape(clean_inputs.shape[0],1,1,-1)
                clean_outputs = clean_outputs.reshape(clean_outputs.shape[0],1,1,-1)                
                
                generated_inputs = generated_inputs.reshape(generated_inputs.shape[0],1,1,-1)
                generated_outputs = generated_outputs.reshape(generated_outputs.shape[0],1,1,-1)                
                
                avg_inputs = (generated_inputs + inputs)/2
                avg_outputs = (generated_outputs+outputs)/2
                
                if self.preconditioner:
                    inputs,outputs = self._precondition(inputs,outputs)
                    clean_inputs,clean_outputs = self._precondition(clean_inputs,clean_outputs)
                    generated_inputs,generated_outputs = self._precondition(generated_inputs,generated_outputs)
                    avg_inputs,avg_outputs = self._precondition(avg_inputs,avg_outputs)
                # set hidden unit activations
                for l in range(self.num_layers):
                    self.slstm[l].net.blobs['h_init_i_jm1'].data[:] = hiddens[l][i, j - 1]
                    self.slstm[l].net.blobs['h_init_im1_j'].data[:] = hiddens[l][i - 1, j]
                    self.slstm[l].net.blobs['c_init_i_jm1'].data[:] = memory[l][i, j - 1]
                    self.slstm[l].net.blobs['c_init_im1_j'].data[:] = memory[l][i - 1, j]

                # compute hidden unit activations
                # print 'hidden activation'
                activations = (inputs) #Giving an average as input

                for l in range(self.num_layers):
                    activations = self.slstm[l].forward(activations)

                # store hidden unit activations
                for l in range(self.num_layers):
                    hiddens[l][i, j] = self.slstm[l].net.blobs['outputs'].data.copy()
                    memory[l][i, j] = self.slstm[l].net.blobs['c_0_0'].data.copy()
                    
                h_norm[i,j] = norm(hiddens[-1][i,j])
                c_norm[i,j] = norm(memory[-1][i,j])
                tanh_c[i,j] = sum(sum(tanh(memory[-1][i,j])))
                #hid_arr[i].append(hiddens[-1][i,j])
                #mem_arr[i].append(memory[-1][i,j])

                #outputs = None
                if method == 'mode':
                    #max_logq = -10000.0
                    avg_f = np.zeros(sample)
                    Y_flat = copy(outputs.reshape(-1,self.num_channels).T) #Storing_yi                    
                    X_flat = copy(outputs.reshape(-1,self.num_channels).T)
                    #print 'Y_flat',Y_flat                    
                    
                    #print 'it',it
                    
                    # print 'Hiddens shape',hiddens[-1][i, j].reshape(-1, self.num_hiddens).T.shape

                    # print 'Inputs shape',inputs.reshape(-1, 12).T.shape
                    H_flat = hiddens[-1][i, j].reshape(-1, self.num_hiddens).T
                    
                    
                    #Sample from posterior
                    # component gate probabilities
                    square_weights = np.square(weights)
                    feature_input = np.dot(features.T,H_flat)
                    gate_energy = np.dot(square_weights,np.square(feature_input))
                    gate_energy = priors - 0.5* np.multiply(np.exp(scales), gate_energy)
                    prior_prob = np.exp(gate_energy)
                    prior_prob = prior_prob/prior_prob.sum()
                        

                    #Computing expert prob for loglik
                    expert_means = np.dot(predictors,H_flat)

                    for k in range(self.mcgsm.num_scales-1):
                        expert_means = np.append(expert_means,np.dot(predictors,H_flat), axis = 1)

                    #Computing posterior
                    #1 dimensional output
                    if 1:#pre_out.shape[0]==1:
                        noise_std = whitened_noise_std*np.ones_like(expert_std)
                        noise_mean = Y_flat.reshape(1)*np.ones_like(expert_std)
                        var_p = (noise_std**2 * expert_std**2)/(noise_std**2+expert_std**2)
                        mean_p = var_p*(expert_means/(expert_std**2) + noise_mean/(noise_std**2))
                        scales_p = 1/np.sqrt(2*np.pi*(noise_std**2+expert_std**2)) * np.exp(-1*(noise_mean-expert_means)**2/(2*(noise_std**2+expert_std**2)))               
                    #else:
                    
                    g = mixture.GMM(n_components = self.mcgsm.num_scales*self.mcgsm.num_components)
                    g.means_ = np.array(mean_p).reshape(-1,1)                        
                    g.covars_ = (var_p).reshape(-1,1)
                    g.weights_= (prior_prob*scales_p).reshape(1,-1)   
                    
                    X_samp = g.sample(n_samples = sample)
                    X_samp[np.isnan(X_samp)] = 1000
                    X_samp[X_samp>1000] = 1000
                    X_samp[X_samp<-1000] = -1000
                    X_samp = X_samp.astype('float64')
                    #print X_samp
                    X_flat = X_samp[np.argmax(g.score(X_samp))]
                    #X_flat =  np.dot(prior_prob.reshape(1,-1),g.means_)
                    outputs = X_flat.reshape(1,1,1,1)
                   # for it in range(sample):      
                   #      print X_flat
                        
                   #     samp =  g.sample()
                   #     print 'samp',samp
                   #     samp1 = copy(samp)
                   #      logl = g.score(samp1)
                   #      print 'logl',logl
                   #      print 'samp now',samp
                   #     if logl > max_logq:
                   #         max_logq = logl
                   #         samp1 = samp
                    
                   #     X_flat = self.mcgsm.sample(H_flat)
                   #     X_flat = copy(g.sample(n_samples = 1,random_state = 1))
                   #     print 'X_flat',  X_flat                        #print g.sample()
                   #     logq = self.mcgsm.loglikelihood(H_flat,X_flat)
                   #     #logl = g.score(X_flat)
                       
                       
                   #     #log_p_n = scipy.stats.norm.logpdf(X_flat.reshape(1),Y_flat.reshape(1),whitened_noise_std)
                   #     #print 'log_p_n' ,log_p_n
                   #     #print 'Y_flat',Y_flat 
                   #     #print 'X_flat',X_flat
                   #     #print 'logq+log_p_n' , logq + log_p_n                       
                   #     #print 'logq', logq
                   #     #print 'logl' , logl                       
                   #     avg_f[it] = logq                      
                       
                       
                   #     if logq > max_logq:
                   #         max_logq = logq
                   #         #print logq
                   #         outputs = copy(X_flat)
                    #print 'Outputs' ,outputs
                    if g.score(X_flat.reshape(-1,1)).reshape(1) > 10:
                        X_flat1 = copy(X_flat)
                        #print 'Before',g.score(X_flat.reshape(-1,1)).reshape(1)
                        #print 'Adding mean'
                        #X_flat =  np.dot(g.weights_,g.means_)                       
                        X_flat = Y_flat                        
                       # if g.score(X_flat.reshape(-1,1))<g.score(X_flat1.reshape(-1,1)): 
                       #     outputs = X_flat.reshape(1,1,1,1)
                       # else:
                       #     X_flat = Y_flat
                        outputs = X_flat.reshape(1,1,1,1)
                        #print 'After',g.score(X_flat.reshape(-1,1)).reshape(1)

                    avg_logl[i].append(g.score(X_flat.reshape(-1,1)).reshape(1))                        
                elif method =='max' :
                    #l_r = 0.01     #0.01,20  0.0001,30
                    #sample = 30
                    
                    avg_f = np.zeros(sample)
                    Y_flat = None
                    H_flat = hiddens[-1][i, j].reshape(-1, self.num_hiddens).T
                    for it in range(sample):
                        print it
                        
                       # if it > 0:                        
                       #     ########Forward_pass                                    
                       #     activations = inputs
                       #     for l in range(self.num_layers):
                       #         activations = self.slstm[l].forward(activations)
                       #                         # store hidden unit activations
                       #     for l in range(self.num_layers):
                       #         hiddens[l][i, j] = self.slstm[l].net.blobs['outputs'].data.copy()
                       #         memory[l][i, j] = self.slstm[l].net.blobs['c_0_0'].data.copy()
                            
                        
                        
                        if it == 0 :
                            Y_flat = copy(outputs.reshape(-1,self.num_channels).T) #Storing noisy inputs

                            # component gate probabilities
                            square_weights = np.square(weights)
                            feature_input = np.dot(features.T,H_flat)
                            gate_energy = np.dot(square_weights,np.square(feature_input))
                            gate_energy = priors - 0.5* np.multiply(np.exp(scales), gate_energy)
                            prior_prob = np.exp(gate_energy)
                            prior_prob = prior_prob/prior_prob.sum()
                                
        
                            #Computing expert prob for loglik
                            expert_means = np.dot(predictors,H_flat)
                            for k in range(self.mcgsm.num_scales-1):
                                expert_means = np.append(expert_means,np.dot(predictors,H_flat), axis = 1)
        
                            #Computing posterior
                            # noise_std = whitened_noise_std*np.ones_like(expert_std)
                            # noise_mean = Y_flat.reshape(1)*np.ones_like(expert_std)
                            # var_p = (noise_std**2 * expert_std**2)/(noise_std**2+expert_std**2)
                            # mean_p = var_p*(expert_means/(expert_std**2) + noise_mean/(noise_std**2))
                            # scales_p = 1/np.sqrt(2*np.pi*(noise_std**2+expert_std**2)) * np.exp(-1*(noise_mean-expert_means)**2/(2*(noise_std**2+expert_std**2)))                        
                            # g = mixture.GMM(n_components = self.mcgsm.num_scales*self.mcgsm.num_components)
                            # g.means_ = np.array(mean_p).reshape(-1,1)                        
                            # g.covars_ = (var_p).reshape(-1,1)
                            # g.weights_= (prior_prob*scales_p).reshape(1,-1)   
                            
                            #X_flat = np.dot(g.weights_,g.means_).reshape(-1,self.num_channels).T
                            X_flat = copy(outputs.reshape(-1,self.num_channels).T) #Storing noisy inputs                            
                            
                        ##########Backward pass
                        # form inputs to MCGSM
                        #compute gradeints
                        df_dh, df_dy, loglik = self.mcgsm._data_gradient(H_flat, X_flat) #Grad of log(p(xi|x<i))
                        df_dh = np.array([df_dh.T.reshape(*hiddens[-1][i,j].shape)]) / H_flat.shape[1]
                        df_dy = df_dy.T.reshape(*outputs.shape) / H_flat.shape[1]
                        
                        # average log-likelihood
                        f = sum(loglik) / H_flat.shape[1]
                        #log_p_n =scipy.stats.norm.logpdf((Y_flat-X_flat).reshape(1),0,scipy.linalg.sqrtm(wht_cov))                        
                        
                        avg_f[it] = f#+log_p_n
                        print 'loglik',f

                        #Posterior prob
                        df_dy += np.dot(np.linalg.inv(wht_cov),(X_flat-Y_flat)).reshape(*outputs.shape) #yi = xi +n; p(yi|xi) = p(n);d(logp(n)/x)                     
                                         
                        #if f_prev > f:
                        #    break
    
                        #for l in range(self.num_layers)[::-1]:
                        #    df_dh = self.slstm[l].backward(df_dh, force_backward=True)['inputs']
                        #inputs += lr*df_dh
                        X_flat += (lr*df_dy).reshape(-1,self.num_channels).T #Estimate of xi updated
                        #print it
                    #print X_flat
                    outputs = X_flat
                    avg_logl[i].append(avg_f)
                    #p_exp_arr[i].append(loglikim)
                elif method == 'wiener' :
                    Y_flat = copy(outputs.reshape(-1,self.num_channels).T)
                    H_flat = hiddens[-1][i, j].reshape(-1, self.num_hiddens).T

                    # component gate probabilities
                    square_weights = np.square(weights)
                    feature_input = np.dot(features.T,H_flat)
                    gate_energy = np.dot(square_weights,np.square(feature_input))
                    gate_energy = priors - 0.5* np.multiply(np.exp(scales), gate_energy)
                    prior_prob = np.exp(gate_energy)
                    prior_prob = prior_prob/prior_prob.sum()
                    
                    #expert means                   
                    expert_means = np.dot(predictors,H_flat)
                    for k in range(self.mcgsm.num_scales-1):
                        expert_means = np.append(expert_means,np.dot(predictors,H_flat), axis = 1)
                    
                    #p(c,s|y_i,h_i)
                    noise_std = whitened_noise_std*np.ones_like(expert_std)
                    noise_mean = Y_flat.reshape(1)*np.ones_like(expert_std)
                    var_p = (noise_std**2 * expert_std**2)/(noise_std**2+expert_std**2)
                    mean_p = var_p*(expert_means/(expert_std**2) + noise_mean/(noise_std**2))
                    scales_p = 1/np.sqrt(2*np.pi*(noise_std**2+expert_std**2)) * np.exp(-1*(noise_mean-expert_means)**2/(2*(noise_std**2+expert_std**2)))
                    pi_cs =(prior_prob*scales_p)
                    #print pi_cs

                    #argmax c,s
                    cs_max=np.argmax(pi_cs)
                    [c_max,s_max]=np.unravel_index(cs_max,pi_cs.shape)
                    #y = mu_c_max,s_max
                    #print 'max component',c_max,s_max
                    #print 'scale', scales_p[c_max,s_max]
                    #print 'mean',mean_p[c_max,s_max]
                    outputs = mean_p[c_max,s_max]*np.ones_like(Y_flat)

                    #print 's_n',whitened_noise_std

                    # for c,s in enumerate(expert_std[k_max,:]) :
                    #     #print 's',s
                    #     mu_k_i =  means[k_max,c]*np.ones_like(Y_flat)
                    #     #print 'mu_k_i',mu_k_i
                    #     #print 'Y_flat',Y_flat
                    #     pi_k_i = prior_prob[k_max,c]/prior_prob[k_max,:].sum()
                    #     #print 'pi_k_i',pi_k_i
                    #     s_n = whitened_noise_std
                    #     outputs += pi_k_i*((s**2)*Y_flat +(s_n**2)*mu_k_i)/(s_n**2+s**2)
                    # print 'diff',Y_flat-outputs
                elif method == 'mean' :
                    Y_flat = copy(outputs.reshape(-1,self.num_channels).T)
                    H_flat = hiddens[-1][i, j].reshape(-1, self.num_hiddens).T

                    # component gate probabilities
                    square_weights = np.square(weights)
                    feature_input = np.dot(features.T,H_flat)
                    gate_energy = np.dot(square_weights,np.square(feature_input))
                    gate_energy = priors - 0.5* np.multiply(np.exp(scales), gate_energy)
                    prior_prob = np.exp(gate_energy)
                    prior_prob = prior_prob/prior_prob.sum()
                    
                    #expert means                   
                    expert_means = np.dot(predictors,H_flat)
                    for k in range(self.mcgsm.num_scales-1):
                        expert_means = np.append(expert_means,np.dot(predictors,H_flat), axis = 1)
                    
                    #p(c,s|y_i,h_i)
                    noise_std = whitened_noise_std*np.ones_like(expert_std)
                    noise_mean = Y_flat.reshape(1)*np.ones_like(expert_std)
                    var_p = (noise_std**2 * expert_std**2)/(noise_std**2+expert_std**2)
                    mean_p = var_p*(expert_means/(expert_std**2) + noise_mean/(noise_std**2))
                    scales_p = 1/np.sqrt(2*np.pi*(noise_std**2+expert_std**2)) * np.exp(-1*(noise_mean-expert_means)**2/(2*(noise_std**2+expert_std**2)))
                    pi_cs =(prior_prob*scales_p)

                    outputs = (pi_cs*mean_p).sum() * np.ones_like(Y_flat)
        
                outputs = outputs.T.reshape(outputs.shape[1], 1, 1, outputs.shape[0])    
                #print 'outputs',outputs       
                #print 'Y_flat',Y_flat


                if self.preconditioner:
                    inputs1, outputs = self._precondition_inverse(inputs, outputs)
                    #inputs2,Y_flat = self._precondition_inverse(generated_inputs,Y_flat)
                
                #print 'whitened diff',Y_flat-outputs  
                
                if method =='max1':
                    for i1 in range(i,i+self.input_mask.shape[0]):
                        for j1 in range(j,j+self.input_mask.shape[1]):
                            if (self.input_mask[i1-i,j1-j]):                            
                               images[:,i1,j1] = inputs[:,:,:,(i1-i)*(self.input_mask.shape[1])+j1-j]   #Check shape                 
                
                
                if max_values is not None:
                    outputs[outputs > max_values] = max_values[outputs > max_values]
                if min_values is not None:
                    outputs[outputs < min_values] = min_values[outputs < min_values]        
                              
                # if self.output_mask.ndim > 2:
                #      images[:, i + i_off, j + j_off][:, self.output_mask[i_off, j_off]] = outputs
                # else:
                #     images[:, i + i_off, j + j_off] = outputs
                              
                #print (sampled_images-images).sum()
                #print outputs
                #print outputs.shape
                #print 'Position to print',i+i_off,j+j_off
                # insert sampled pixels into images
                if self.output_mask.ndim > 2:
                    sampled_images[:, i + i_off, j + j_off][:, self.output_mask[i_off, j_off]] = outputs
                else:
                    sampled_images[:, i + i_off, j + j_off] = outputs
 
                #print 'Final diff',images[:,i+i_off,j+j_off]-sampled_images[:, i + i_off , j + j_off]
                #print (sampled_images-images).sum()
        sampled_images = sampled_images.reshape(*shape)
        images = images.reshape(*shape)
        
        #np.save('p_exp.npy',p_exp_arr)
        #np.save('hiddens.npy',hid_arr)
        #np.save('memory.npy',mem_arr)
        #np.save('avg_logl.npy',avg_logl)
        #sampled_images = sampled_images.reshape(*shape)
        sample_act = {'h':h_norm,'c':c_norm,'t':tanh_c}
        savemat('sample_act.mat',sample_act)
        return images,sampled_images,np.array(avg_logl)

    def denoise_4dir(self,
            images,
            min_values=None,
            max_values=None,
            mask=None,
            return_loglik=False,
            method = 'max',
            sample = 10,
            lr=0.01,
            sigma=0.08,
            clean_images = None,
            clean_images_arr  =None,
            num_dir=4):
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

        images_arr = [images[::(-1)**i,::(-1)**(i/2)] for i in range(num_dir)]
        #clean_images_arr = [clean_images[::(-1)**i,::(-1)**(i/2)] for i in range(num_dir)]

        for i in range(num_dir):
            # reshape images into four-dimensional arrays
            shape = images_arr[i].shape
            if images_arr[i].ndim == 2:
                images_arr[i] = images_arr[i][None, :, :, None]
            elif images_arr[i].ndim == 3:
                if self.num_channels > 1:
                    images_arr[i] = images_arr[i][None]
                else:
                    images_arr[i] = images_arr[i][:, :, :, None]
            
            if clean_images_arr[i].ndim == 2:
                clean_images_arr[i] = clean_images_arr[i][None, :, :, None]
            elif clean_images_arr[i].ndim == 3:
                if self.num_channels > 1:
                    clean_images_arr[i] = clean_images_arr[i][None]
                else:
                    clean_images_arr[i] = clean_images_arr[i][:, :, :, None]

        sampled_images = copy(images_arr[0])

        for i_off, j_off in zip(
                range(self.output_mask.shape[0]),
                range(self.output_mask.shape[1])):
            if any(self.output_mask[i_off, j_off]):
                break

        #distribution params
        priors = np.array(self.mcgsm.priors)
        scales = self.mcgsm.scales
        weights = self.mcgsm.weights
        features = self.mcgsm.features
        cholesky_factors = self.mcgsm.cholesky_factors
        predictors = np.array(self.mcgsm.predictors)
        predictors = predictors.reshape(self.mcgsm.num_components,-1)
        expert_std = np.sqrt(1.0/np.exp(scales))
        pre_in = self.preconditioner.pre_in # size nbd x nbd
        #mean_out = self.preconditioner.mean_out # size  1,1
        #print mean_out
        pre_out = self.preconditioner.pre_out # size 1,1
        #print pre_out
        predictor = self.preconditioner.predictor            
        b = pre_out * np.dot(predictor,pre_in)
        b = np.square(b)
        whitened_noise_std = sigma * np.sqrt(pre_out**2 + b.sum())
        print 'whitened_sigma',whitened_noise_std

        inputs_arr = []
        outputs_arr = []
        hiddens_arr = []
        loglik_arr = []

        for i,images in enumerate(images_arr):  #Change to images_arr
            inputs,outputs = self._preprocess(images)
            
            logjacobian = self.preconditioner.logjacobian(
                inputs.reshape(-1, sum(self.input_mask)).T,
                outputs.reshape(-1, self.num_channels).T)

            inputs, outputs = self._precondition(inputs, outputs)
            
            #Computing hidden units
            hiddens = inputs
            batch_size = min([hiddens.shape[0], 32])

            for l in range(self.num_layers):
                # create SLSTM
                print 'creating lstm layer', l
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

            # evaluate log-likelihood
            loglik = self.mcgsm.loglikelihood(
                hiddens.reshape(-1, self.num_hiddens).T,
                outputs.reshape(-1, self.num_channels).T) + logjacobian
            loglik = loglik.reshape(hiddens.shape[0],hiddens.shape[1],hiddens.shape[2])
            
            loglik = loglik[:,:-2,:]
            loglik = loglik[:,::(-1)**i,::(-1)**(i/2)]
            hiddens = hiddens[:,:-2,:,:]
            hiddens=hiddens[:,::(-1)**i,::(-1)**(i/2),:]
            inputs = inputs[:,:-2,:,:]
            inputs = inputs[:,::(-1)**i,::(-1)**(i/2),:]
            outputs = outputs[:,:-2,:,:]
            outputs = outputs[:,::(-1)**i,::(-1)**(i/2),:]

            loglik_arr.append(loglik)
            hiddens_arr.append(hiddens)
            inputs_arr.append(inputs)
            outputs_arr.append(outputs)

        loglik_arr = np.asarray(loglik_arr)
        hiddens_arr = np.asarray(hiddens_arr)
        inputs_arr = np.asarray(inputs_arr)
        outputs_arr = np.asarray(outputs_arr)
        #print 'inputs arr' ,inputs_arr.shape
        #max_h= np.stack([np.argmax(loglik_arr,axis=0)]*self.num_hiddens,axis=-1)
        #print max_h[:,:,:,0]
        #print max_h.shape
        #hiddens_max = np.choose(max_h,hiddens_arr)
        #print hiddens_max.shape
        print outputs_arr.shape

        #print images.shape
        for i in range(images_arr[0].shape[1] - self.input_mask.shape[1] + 1):
            for j in range(images_arr[0].shape[2] - self.input_mask.shape[1] + 1):

                Y_flat_arr = [o[:,i,j].reshape(-1,self.num_channels).T for o in outputs_arr]
                H_flat_arr = [h[:,i,j].reshape(-1, self.num_hiddens).T for h in hiddens_arr]
                #H_flat_arr = [hiddens_max[:,i,j].reshape(-1, self.num_hiddens).T]

                # component gate probabilities
                pi_cs_arr = []
                mean_p_arr = []

                for H_flat,Y_flat in zip(H_flat_arr,Y_flat_arr):
                    square_weights = np.square(weights)
                    feature_input = np.dot(features.T,H_flat)
                    gate_energy = np.dot(square_weights,np.square(feature_input))
                    gate_energy = priors - 0.5* np.multiply(np.exp(scales), gate_energy)
                    prior_prob = np.exp(gate_energy)
                    prior_prob = prior_prob/prior_prob.sum()

                    #expert means                   
                    expert_means = np.dot(predictors,H_flat)
                    for k in range(self.mcgsm.num_scales-1):
                        expert_means = np.append(expert_means,np.dot(predictors,H_flat), axis = 1)
                
                    #p(c,s|y_i,h_i)
                    noise_std = whitened_noise_std*np.ones_like(expert_std)
                    noise_mean = Y_flat.reshape(1)*np.ones_like(expert_std)
                    var_p = (noise_std**2 * expert_std**2)/(noise_std**2+expert_std**2)
                    mean_p = var_p*(expert_means/(expert_std**2) + noise_mean/(noise_std**2))
                    scales_p = 1/np.sqrt(2*np.pi*(noise_std**2+expert_std**2)) * np.exp(-1*(noise_mean-expert_means)**2/(2*(noise_std**2+expert_std**2)))
                    pi_cs =(prior_prob*scales_p)

                    pi_cs_arr.append(pi_cs)
                    mean_p_arr.append(mean_p)

                pi_cs_arr = np.asarray(pi_cs_arr)
                mean_p_arr = np.asarray(mean_p_arr)
                #argmax c,s
                dcs_max=np.argmax(pi_cs_arr)
                [d_max,c_max,s_max]=np.unravel_index(dcs_max,pi_cs_arr.shape)
                
                #y = mu_c_max,s_max
                #print 'max component',c_max,s_max
                #print 'scale', scales_p[c_max,s_max]
                #print 'mean',mean_p[c_max,s_max]
                outputs = mean_p_arr[d_max,c_max,s_max]*np.ones_like(Y_flat)

                outputs = outputs.T.reshape(outputs.shape[1], 1, 1, outputs.shape[0]) 
                if self.preconditioner:
                    #print 'inputs shape',inputs.shape
                    inputs1, outputs = self._precondition_inverse(inputs_arr[d_max,:,i,j,:].reshape(1,1,1,-1), outputs)
                    inputs2,Y_flat = self._precondition_inverse(inputs_arr[d_max,:,i,j,:].reshape(1,1,1,-1),Y_flat)
                
                sampled_images[:, i + i_off , j + j_off] =  outputs 

        sampled_images = sampled_images.reshape(*shape)
        images = images_arr[0].reshape(*shape)
        avg_logl=0
        return images,sampled_images,np.array(avg_logl)

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
