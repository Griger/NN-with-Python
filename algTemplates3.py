import numpy as np
import time
import theano.tensor as T
from theano import function, config, scan, shared
from constants import *

np.random.seed(12345678)

lr = 0.1

imgs = np.load("data/trainImg.npy")
lbls = np.load("data/binTrainLbl.npy")
imgsTest = np.load("data/testImg.npy")
lblsTest = np.load("data/testLbl.npy")

x = T.vector('x')
y = T.vector('y')
A = T.matrix('A')

w = shared(np.random.rand(ROWS*COLS, 10).astype(config.floatX))

out = T.dot(x, w)/(28.0*28.0)
err = y - out
gw, aux = scan(lambda scalar,x: scalar * x, sequences=[err], non_sequences=x)

train = function([x, y], err, updates = {w: w + (lr * gw.T)})

#Train
t0 = time.time()
for img, lbl in zip(imgs, lbls):
    train(img, lbl)
t1 = time.time()

dot = function([x,A], T.dot(x,A))
#Test
predicted = []
nError = 0
for img, lbl in zip(imgsTest, lblsTest):
    p = np.argmax(dot(img, w.get_value()))
    predicted.append(p)
    if p != lbl:
        nError += 1

print("Tiempo de entrenamiento:", t1-t0, "segundos.\nSe han cometido", nError, "errores. Un", nError/NTEST*100.0, "%")
