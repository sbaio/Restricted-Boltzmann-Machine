## Wed 07 Dec 2016 12:51:49 PM CET 
## Author : SHEN Xi

from loadMNIST import load_mnist
import pylab as pl
import numpy as np

## image is dimension 28 * 28
images, labels = load_mnist('testing', digits=[2], path = 'data/')
pl.imshow(images.mean(axis=0), cmap='gray')
pl.show()

def sigmoid(eta):
    '''Return the logistic sigmoid function of the argument.'''
return 1. / (1. + np.exp(-eta))


def bernoulli(p):
    '''Return an array of boolean samples from Bernoulli(p).
    Parameters
    ----------
    p : ndarray
        This array should contain values in [0, 1].
    Returns
    -------
    An array of the same shape as p. Each value in the result will be a boolean
    indicating whether a single Bernoulli trial succeeded for the corresponding
    element of `p`.
    '''
return rng.rand(*p.shape) < p

class RBM(object):
	"""Restricted Boltzmann Machine (RBM)  """
	def __init__(
		self,
		n_visible=784,
		n_hidden=500,
		W=None,
		hbias=None,
		vbias=None,
		seed = None,
	):
	"""
	RBM constructor. Defines the parameters of the model along with
	basic operations for inferring hidden from visible (and vice-versa),
	as well as for performing CD updates.

	:param n_visible: number of visible units

	:param n_hidden: number of hidden units

	:param W: None for standalone RBMs or symbolic variable pointing to a
	shared weight matrix in case RBM is part of a DBN network; in a DBN,
	the weights are shared between RBMs and layers of a MLP

	:param hbias: None for standalone RBMs or symbolic variable pointing
	to a shared hidden units bias vector in case RBM is part of a
	different network

	:param vbias: None for standalone RBMs or a symbolic variable
	pointing to a shared visible units bias
	"""

		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if seed is None:
			# create a number generator
			numpy_rng = np.random.RandomState(1234)

		if W is None:
			# W is initialized with uniformely
			# sampled from -4*sqrt(6./(n_visible+n_hidden)) and
			# 4*sqrt(6./(n_hidden+n_visible)) the output of uniform 
			W = np.asarray(
			numpy_rng.uniform(
				low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
				high=4 * np.sqrt(6. / (n_hidden + n_visible)),
				size=(n_visible, n_hidden)
			),
			)
			
		if hbias is None:
			# create variable for hidden units bias
			hbias = np.zeros(n_hidde, 1)

		if vbias is None:
			# create variable for visible units bias
			vbias = np.zeros(n_visible, 1)

		self.W = W
		self.hbias = hbias
		self.vbias = vbias

	def propup(self, vis):
	'''This function propagates the visible units activation upwards to
	the hidden units

	Note that we return also the pre-sigmoid activation of the
	layer. As it will turn out later, due to how Theano deals with
	optimizations, this symbolic variable will be needed to write
	down a more stable computational graph (see details in the
	reconstruction cost function)

	'''
		h_pre_sigmoid_activation = vis.dot(self.W) + self.hbias
	return h_pre_sigmoid_activation
















