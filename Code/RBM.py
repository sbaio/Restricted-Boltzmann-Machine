## Wed 07 Dec 2016 12:51:49 PM CET 
## Author : SHEN Xi

from loadMNIST import load_mnist
import pylab as pl
import numpy as np
import os
import time
import argparse
import json



def sigmoid(eta):
    '''Return the logistic sigmoid function of the argument.'''
    return 1. / (1. + np.exp(-eta))

def soft_plus(x):
	'''Return the soft plus function of the argument.
	a soft approximation of max(0, x)
	'''
	return np.log(1 + np.exp(x))


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
	return np.random.rand(*p.shape) < p

class RBM(object):
	"""Restricted Boltzmann Machine (RBM)  """
	def __init__(
		self,
		n_visible=784,
		n_hidden=500,
		learning_rate = 0.001,
		k = 1,
		batchsize = 100,
		max_epoch = 100,
		val_ratio = 0.2,
		output_file = "log_training.txt",
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
		:param learning_rate: step of gradient descent
		:param k: iteration number of Gibbs sampling 
		:param batchsize: batchsize for train
		:param max_epoch: maximum epochs during the train
		:param val_ratio: ratio of validation set  
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
		output_file = os.path.abspath(output_file)
		if os.path.exists(output_file):
			print 'Training information will be written into %s...\n'%output_file
		else:
			print '%s doesnt exist, it will be created to supervise the training process'%output_file

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
			hbias = np.zeros((n_hidden, 1), dtype = np.float32)

		if vbias is None:
			# create variable for visible units bias
			vbias = np.zeros((n_visible, 1), dtype = np.float32)

		if k < 1:
			raise RuntimeError('The iteration of Gibbs sampling k should be >= 1')

		self.learning_rate = learning_rate
		self.batchsize = batchsize
		self.max_epoch = max_epoch
		self.k = k 
		self.val_ratio = val_ratio
		self.output_file = output_file
		self.W = W
		self.hbias = hbias
		self.vbias = vbias

	def hidden_sigmoid_activation(self, vis_state):
		'''This function propagates the visible units activation upwards to
		the hidden units, and only compute the values JUST AFTER the sigmoid function.
		It is useful for the gradient descent.
		## vis : visible layer, each node is a pixel
		## vis : n_visible * 1
		'''
		return sigmoid(self.W.T.dot(vis_state) + self.hbias)

	def hidden_expectation(self, vis_state):
		'''This function propagates the visible units activation upwards to
		the hidden units, and compute directily the hidden units
		## vis : visible layer, each node is a pixel
		## vis : n_visible * 1
		'''
		h_sigmoid_activation = self.hidden_sigmoid_activation(vis_state)

		return bernoulli(h_sigmoid_activation)


	def visible_expectation(self, hidden_state):
		'''This function propagates the hidden units activation upwards to
		the visiable units
		## hidden : hidden layer, each node is a pixel
		## hidden : n_hidden * 1
		'''
		vis_pre_sigmoid_activation = self.W.dot(hidden_state) + self.vbias
		vis_state = sigmoid(vis_pre_sigmoid_activation)

		return bernoulli(vis_state)

	def log_proba(self, vis_state):
		'''This function computes the maginal probability of visible state p(visible_state)	'''
		term1 = np.sum(soft_plus(self.hbias + (self.W.T).dot(vis_state)))
		term2 =  (vis_state.T).dot(self.vbias)
		return  (term1 + term2)[0][0]


	def marginal_proba(self, vis_state):
		'''This function computes the log probability '''
		return np.exp(self.log_proba(vis_state))

	def gibbs_sampling_from_vis_state(self, vis_state):
		'''This function generates a new sample (hidden state + visible state) with a given visible state'''
		
		h_sigmoid_activation = self.hidden_sigmoid_activation(vis_state)
		new_hidden_state = bernoulli(h_sigmoid_activation)
		new_vis_state = self.visible_expectation(new_hidden_state)
		
		return (new_hidden_state, new_vis_state, h_sigmoid_activation)

	def gibbs_sampling_from_hidden_state(self, hidden_state):
		'''This function generates a new sample (hidden state + visible state) with a given hidden state'''
		
		new_vis_state = self.visible_expectation(hidden_state)
		new_hidden_state = self.hidden_expectation(new_vis_state)
		
		return (new_hidden_state, new_vis_state)

	def estimate_dist_real_gibbs(self, dataset):
		'''This function provides another measure for the algo
			it generates the gibbs sampling from the dataset, then calculate the distance 
			between the real image and the image from gibbs
		'''
		(n_image, n_visible) = dataset.shape
		(new_hidden_state, new_vis_state, h_sigmoid_activation) = self.gibbs_sampling_from_vis_state(dataset.T)
		dist = np.linalg.norm(new_vis_state - dataset.T)**2/n_image

		return  dist
		
	def gradient_log_proba(self, vis_state): 
		'''This function compute gradient for a sample, it returns the following parameters with their dimensions info : 
	
		# d_W : n_visible * n_hidden
		# d_hbias : n_hidde *  1
		# d_vbias : n_visible * 1
		'''
		(new_hidden_state, new_vis_state, h_sigmoid_activation) = self.gibbs_sampling_from_vis_state(vis_state)
		for i in range(self.k-1):
			(new_hidden_state, new_vis_state, new_h_sigmoid_activation) = self.gibbs_sampling_from_vis_state(new_vis_state)
		
		new_h_sigmoid_activation = self.hidden_sigmoid_activation(new_vis_state)
		dW = h_sigmoid_activation.dot(vis_state.T) - new_h_sigmoid_activation.dot(new_vis_state.T)
		dhbias = h_sigmoid_activation - new_h_sigmoid_activation
		dvbias = vis_state - new_vis_state

		return (dW, dhbias, dvbias)

	def train(self, dataset, outputJson):
		'''This function trains the input dataset, for the dataset, it will be at first SHUFFLED and separate into different batch
	
		# dataset :  n_images * n_visible, each image is in a row vector  
		'''
		(n_image, n_visible) = dataset.shape
		np.random.permutation(dataset)
		n_validation = int(n_image * self.val_ratio)
		trainset  = dataset[n_validation :]
		validset = dataset[: n_validation ]
		n_batch = (n_image - n_validation)/self.batchsize + 1

		log_file = open(self.output_file, 'w')
		log_file.write("\tNumber of Epoch\tTraining -Dist\tValidation - Dist\tTraining Time(CPU time/s)\tValidation Time(CPU time/s)\n")
		current_epoch = 1
		final_weight = {'W' : [], 'hidden_bias' : [], 'visible_bias' : [], 'visible_dim' : self.n_visible, 'hidden_dim' : self.n_hidden, 'train_loss' : 0, 'valid_loss' : np.inf, 'batchsize' : self.batchsize, 'learning_rate': self.learning_rate, 'k' : self.k}
		print 'at epoch 0, training loss is %.5f, validation loss is %.5f, using time : %.5f'%(self.estimate_dist_real_gibbs(trainset), self.estimate_dist_real_gibbs(validset), 0)
		while current_epoch < self.max_epoch + 1 :
			log_weight = open(outputJson,  'w')
			start_time = time.clock()
			log_proba_epoch = 0
			np.random.permutation(trainset)
			for i in range(n_batch):
				batchsize = self.batchsize if i <  n_batch - 1 else len(trainset) -  (n_batch - 1) * self.batchsize
				if batchsize == 0 :
					continue
				
				(dW, dhbias, dvbias) = self.gradient_log_proba(trainset[(i) * self.batchsize : min((i+1) * self.batchsize, len(trainset))].T)
				dW_batch =  dW.T/batchsize
				dhbias_batch = dhbias.mean(1).reshape((-1,1))
				dvbias_batch = dvbias.mean(1).reshape((-1,1))
				#print 'at epoch %d batch %d, negative training log proba is %.5f'%(current_epoch, i, self.estimate_dist_real_gibbs(trainset))
				#log_proba_batch = log_proba_batch/batchsize
				#print 'at epoch %d batch %d, negative training log proba is %.5f'%(current_epoch, i, log_proba_batch)

				##update the parameters 
				self.W = self.W + self.learning_rate * dW_batch
				self.hbias = self.hbias + self.learning_rate * dhbias_batch
				self.vbias = self.vbias + self.learning_rate * dvbias_batch
			#log_train = np.mean([-self.log_proba(vis_state.reshape((-1, 1))) for vis_state in trainset])
			log_train = self.estimate_dist_real_gibbs(trainset) 
			traintime = time.clock()
			log_val = self.estimate_dist_real_gibbs(validset) 
			validtime = time.clock() 
			#log_val = np.mean([-self.log_proba(vis_state.reshape((-1, 1))) for vis_state in validset])
			#validtime = time.clock() 
			
			if log_val < final_weight['valid_loss'] : 
				final_weight['W'] = self.W.tolist()
				final_weight['hidden_bias'] = self.hbias.tolist()
				final_weight['visible_bias'] = self.vbias.tolist()
				final_weight['train_loss'] = log_train
				final_weight['valid_loss'] = log_val	
				with open(outputJson, 'w') as outfile:
					json.dump(final_weight, outfile, separators=(',', ':'), indent = 2)
			#print 'at epoch %d, training negative log proba is %.5f, validation negative log proba is %.5f, using time : %.5f'%(current_epoch, log_train, log_val, validtime - start_time)
			print 'at epoch %d, training loss is %.5f, validation loss is %.5f, using time : %.5f'%(current_epoch, log_train, log_val, validtime - start_time)
			log_file.write("\t%d\t%.5f\t%.5f\t%.5f\t%.5f\n"%(current_epoch, log_train, log_val, traintime - start_time, validtime - traintime))
			current_epoch += 1
		log_file.close()
		
				



	
if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	print '******--------- Training RBM on MNIST ------*******'
	# global setup settings, and checkpoints
	parser.add_argument('-hi', '--hidden', dest='hidden', type=int, default='100', help='Number of hidden units in the hidden layer')
	parser.add_argument('-l', '--learning_rate', dest='lrate', type=float, default=0.001, help='The step of gradient descent')
	parser.add_argument('--iteration_CD', dest='iter_CD', default='1', type=int, help='Number of iterations in the CD algorithm')
	parser.add_argument('-o', '--outputfile', dest='output', type=str, default='log_training.txt', help='Output file for the training supervision')
	parser.add_argument('-j', '--outputJson', dest='outjson', type=str, default='weight.json', help='Output file store the final coefficient')
	parser.add_argument('--batchsize', dest='bsize', type=int, default='100', help='Number of training sample in a single batch')
	parser.add_argument('--max_epoch', dest='mepoch', type=int, default=100, help='Number of maximum epochs during the train')
	parser.add_argument('--val_ratio', dest='vratio',  type=float, default=0.2, help='Ratio of validation set')

	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print 'parsed parameters:'
	print json.dumps(params, indent = 2)
	
	## image is dimension n_images * 28 * 28
	images, labels = load_mnist('training', digits=np.arange(10), path = '../Data/')
	#pl.imshow(images.mean(axis=0), cmap='gray')
	#pl.show()
	(n_images, img_width, img_height) = images.shape
	dataset = np.zeros((n_images, img_width * img_height))
	for i in range(n_images) : 
		dataset[i] = np.array(np.squeeze(images[i, :, :].reshape((-1,1))) > 0, dtype = np.float32)

	model = RBM(n_visible=img_width * img_height,
				n_hidden = params['hidden'],
				learning_rate = params['lrate'],
				k = params['iter_CD'],
				batchsize = params['bsize'],
				max_epoch = params['mepoch'],
				val_ratio = params['vratio'],
				output_file = params['output']
				)
	outputjson = params['outjson']
	model.train(dataset, outputjson)