from loadMNIST import load_mnist
import pylab as pl

## image is dimension 28 * 28
images, labels = load_mnist('testing', digits=[2], path = 'data/')
pl.imshow(images.mean(axis=0), cmap='gray')
pl.show()


