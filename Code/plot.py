import numpy as np
import pylab as pl 
f = open('log_training.txt', 'r')
data = f.read()
data = data.split('\n')
data = data[1:-1]
 
 

epoch = []

train_loss = []

valid_loss = []

train_time = []

valid_time = []

for i in range(len(data)):
     line = data[i].split('\t')[1:]
     epoch.append( int(line[0]))
     train_loss.append(float(line[1]))
     valid_loss.append(float(line[2]))
     train_time.append(float(line[3]))
     valid_time.append(float(line[4]))
     

pl.plot(epoch, train_loss, '--r', lw = 3, label = 'Train loss, ||X - X\'||')
pl.plot(epoch, valid_loss, '--b', lw = 3, label = 'Validation loss, ||X - X\'||')
pl.xlabel('Nb Epochs')
pl.ylabel('||X - X\'||')
pl.legend(loc = 'best')
pl.grid('on')
pl.title('Hidden units = 100, Learning rate = 0.001, 1-CD, batchsize = 100, validation ratio = 0.2, Validation Error = 43.99160')
pl.show()

