import numpy as np
import time
import theano.tensor as T
from theano import function, config, scan, shared
from constants import *

imgs = np.load("data/trainImg.npy")
lbls = np.load("data/trainLbl.npy")

lr = 0.1 #learn rate

w = shared(np.random.normal(size = (ROWS*COLS, 10)).astype(config.floatX))
b = shared(np.zeros(10, dtype = config.floatX))

out = T.dot(sx,w) + b

print("El tipo de estos bichos es:", b)
