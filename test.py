import numpy as np 

train = np.load('data/train.npy')

for i in train:
    i.resize(1,14352)

for i in train:
    print(i.shape)