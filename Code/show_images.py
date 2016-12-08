

from loadMNIST import load_mnist
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def showImage(image):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	imgplot = ax.imshow(image,cmap=mpl.cm.Greys)
	imgplot.set_interpolation('nearest')
	ax.xaxis.set_ticks_position('top')
	ax.yaxis.set_ticks_position('left')
	plt.show()

def show_10_Images(image):
	fig = plt.figure()
	for i in range(10):
		
		ax = fig.add_subplot(2,5,i+1)
		imgplot = ax.imshow(image,cmap=mpl.cm.Greys)
		imgplot.set_interpolation('nearest')
		ax.xaxis.set_ticks_position('top')
		ax.yaxis.set_ticks_position('left')
	plt.show()


def showImages(images):
	# for small number of images
	fig = plt.figure()
	n = len(images)

	for i in range(n):
		ax = fig.add_subplot(1,n,i+1)
		image = images[i]
		imgplot = ax.imshow(image,cmap=mpl.cm.Greys)
		imgplot.set_interpolation('nearest')
		ax.xaxis.set_ticks_position('top')
		ax.yaxis.set_ticks_position('left')
	plt.show()

def plot_10_by_10_images(images):
	""" Plot 100 MNIST images in a 10 by 10 table. Note that we crop
	the images so that they appear reasonably close together.  The
	image is post-processed to give the appearance of being continued."""

	n = images.shape[0]

	q = n // 10
	r = n%10
	print n,q,r

	fig = plt.figure()
	plt.ion()

	for x in range(q):
		print x
		if not x%10:
			plt.clf()
		for y in range(10):
			ax = fig.add_subplot(10, 10, 10*y+x%10+1)
			ax.matshow(images[10*y+x%10], cmap = mpl.cm.binary)
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
		plt.show()
		_=raw_input("Press enter to show next 10")

def generate_random_image():
	# generate random image of type uint8 and size 28*28

	a = np.random.randint(256,size=28*28,dtype='uint8')
	a = a.reshape((28,28))
	return a

def image_to_vector(im):
	b = np.squeeze(im.reshape((-1,1)))/255.
	return b

def vec_to_image(vec):
	b = np.reshape(vec,(28,28))
	return b


images, labels = load_mnist('training', digits=np.arange(10), path = '../Data/')

a = generate_random_image()
#a = images[0]
#b = np.squeeze(a.reshape((-1,1)))/255.

#print b.shape
#print b[:]

showImage(images[0])

#c = vec_to_image(b)
#showImage(c)

#showImages([a,c])

#showImage(d)
#print c.shape

