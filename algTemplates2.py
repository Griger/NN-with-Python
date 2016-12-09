import numpy as np
import time
import theano.tensor as T
from theano import function, config, scan, shared
from constants import *

np.random.seed(12345678)

imgs = np.load("data/trainImg.npy")
lbls = np.load("data/trainLbl.npy")
imgsTest = np.load("data/testImg.npy")
lblsTest = np.load("data/testLbl.npy")

lr = 0.1 #learn rate

#initialize weights in [0,1]
w = np.random.rand(ROWS*COLS, 10).astype(config.floatX)

u = T.vector('u')
X = T.matrix('X')

#declare function that computes a vector matrix product
dot = function([u,X], T.dot(u,X))

#NN training
t0 = time.time()
for img, lbl in zip(imgs, lbls):
    predicted = np.argmax(dot(img, w))
    if (predicted == lbl):
        w[img > 0,predicted] += lr
    else:
        w[img > 0, int(lbl)] += lr
        w[img > 0, predicted] -= lr
t1 = time.time()

print("Training time: ", (t1-t0))

errors = 0
predictedTest = []
for img, lbl in zip(imgsTest, lblsTest):
    predicted = np.argmax(dot(img, w))
    predictedTest.append(predicted)
    if (predicted != lbl):
        errors += 1

print("Test errors:", errors, "Test error rate:", errors/NTEST*100, "%")

errors = 0
predictedTrain = []
for img, lbl in zip(imgs, lbls):
    predicted = np.argmax(dot(img, w))
    predictedTrain.append(predicted)
    if (predicted != lbl):
        errors += 1

print("Train errors:", errors, "Train error rate:", errors/NTRAIN*100, "%")
