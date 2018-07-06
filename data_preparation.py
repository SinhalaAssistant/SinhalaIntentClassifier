import os
import numpy as np
import pandas as pd

def generator(path, batch_size):
    print("Initializing Generator")

    while True:
        train = []
        test = []
        bs = 1
        f = open(path)
        for line in f:
            if (bs <= batch_size):
                x = np.array(line.split(',')).astype(np.float)

                train.append(np.trim_zeros(x[0:186576]).tolist())
                # test.append(x[186576])
                bs += 1
            else:
              return train

        return train
        break

        f.close()


k= generator('data/MFCCFeatures.csv',2000)
# print(k)
np.save('train.npy',k)
# np.savetxt('train.csv',t,delimiter=',')
