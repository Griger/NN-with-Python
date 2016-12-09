import numpy as np
import time
import theano.tensor as T
from theano import function, config, scan, shared
from constants import *

np.random.seed(12345678)

lr = 0.1 #learning rate

imgs = np.load("data/trainImg.npy")
binLbls = np.load("data/binTrainLbl.npy")
lbls = np.load("data/trainLbl.npy")
imgsTest = np.load("data/testImg.npy")
lblsTest = np.load("data/testLbl.npy")

x = T.vector('x')
y = T.vector('y')
A = T.matrix('A')

#initialize weights in [0,1]
w = shared(np.random.rand(ROWS*COLS, 10).astype(config.floatX))

out = T.dot(x, w)/(28.0*28.0)
err = y - out

#gw represents the weight increments
gw, aux = scan(lambda scalar,x: scalar * x, sequences=[err], non_sequences=x)

#function that trains the NN with a train image, update weights
train = function([x, y], err, updates = {w: w + (lr * gw.T)})

#NN training
t0 = time.time()
for img, lbl in zip(imgs, binLbls):
    train(img, lbl)
t1 = time.time()

print("Training time: ", (t1-t0))

#function that computes vector matrix product
dot = function([x,A], T.dot(x,A))

predictedTest = []
errors = 0
for img, lbl in zip(imgsTest, lblsTest):
    p = np.argmax(dot(img, w.get_value()))
    predictedTest.append(p)
    if p != lbl:
        errors += 1
print("Test errors:", errors, "Error rate:", errors*100.0/NTEST)

predictedTrain = []
errors = 0
for img, lbl in zip(imgs, lbls):
    p = np.argmax(dot(img, w.get_value()))
    predictedTrain.append(p)
    if p != lbl:
        errors += 1

print("Train errors:", errors, "Train error rate:", errors/NTRAIN*100, "%")
