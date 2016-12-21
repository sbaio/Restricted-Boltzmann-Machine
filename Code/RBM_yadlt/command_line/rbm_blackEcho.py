import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

import config

from yadlt.models.rbm_models import rbm
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_boolean('encode_train', False, 'Whether to encode and store the training set.')
flags.DEFINE_boolean('encode_valid', False, 'Whether to encode and store the validation set.')
flags.DEFINE_boolean('encode_test', False, 'Whether to encode and store the test set.')
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_string('save_reconstructions', 'Reconstructions/recons.npy', 'Path to a .npy file to save the reconstructions of the model.')
flags.DEFINE_string('save_parameters', 'Parameters/', 'Path to save the parameters of the model.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')

# RBM configuration
flags.DEFINE_integer('num_hidden', 50, 'Number of hidden units.')
flags.DEFINE_string('visible_unit_type', 'bin', 'Type of visible units. ["bin", "gauss"]')
flags.DEFINE_string('main_dir', 'rbm/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('model_name', 'rbm_model', 'Name for the model.')
flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_integer('gibbs_sampling_steps', 1, 'Number of gibbs sampling steps in Contrastive Divergence.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('stddev', 0.1, 'Standard deviation for the Gaussian visible units.')
flags.DEFINE_integer('num_epochs', 3, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')
flags.DEFINE_integer('transform_gibbs_sampling_steps', 20, 'Gibbs sampling steps for the transformation of data.')

assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert FLAGS.cifar_dir != '' if FLAGS.dataset == 'cifar10' else True
assert FLAGS.visible_unit_type in ['bin', 'gauss']

utilities.random_seed_np_tf(FLAGS.seed)

trX, vlX, teX = datasets.load_mnist_dataset(mode='unsupervised')
width, height = 28, 28

models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

# Create the object
r = rbm.RBM(num_hidden=FLAGS.num_hidden, main_dir=FLAGS.main_dir,
            models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir,
            visible_unit_type=FLAGS.visible_unit_type, learning_rate=FLAGS.learning_rate,
            num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, stddev=FLAGS.stddev, verbose=FLAGS.verbose,
            gibbs_sampling_steps=FLAGS.gibbs_sampling_steps, model_name=FLAGS.model_name)

# Fit the model
print('Start training...')
r.fit(trX, teX, restore_previous_model=FLAGS.restore_previous_model)

# Save the model paramenters
# print('Saving the parameters of the model...')
params = r.get_model_parameters()
for p in params:
    np.save(FLAGS.save_parameters + '-' + p, params[p])

# Save the reconstructions of the model

#print('Saving the reconstructions for the test set...')
#np.save(FLAGS.save_reconstructions, r.reconstruct(teX))

recons = r.reconstruct(teX)
images = np.zeros((recons.shape[0],28,28))

for i in range(0,recons.shape[0]):
    images[i] = recons[i,:].reshape((28,28))


def showImage(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

#create loss plot over epochs
print(r.lossOverEpochs)

#for i in range(0,images.shape[0]):
 #   showImage(images[i])

