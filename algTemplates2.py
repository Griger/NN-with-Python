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

w = np.random.rand(ROWS*COLS, 10).astype(config.floatX)

u = T.vector('u')
X = T.matrix('X')

dot = function([u,X], T.dot(u,X))

t0 = time.time()
for img, lbl in zip(imgs, lbls):
    predicha = np.argmax(dot(img, w))
    if (predicha == lbl):
        w[img > 0,predicha] += lr
    else:
        w[img > 0, int(lbl)] += lr
        w[img > 0, predicha] -= lr
t1 = time.time()

print("Hemos tardado: ", (t1-t0))

errores = 0

predichas = []
for img, lbl in zip(imgsTest, lblsTest):
    predicha = np.argmax(dot(img, w))
    predichas.append(predicha)
    if (predicha != lbl):
        errores += 1

print("Se han obtenido", errores, "errores. Un", errores/NTEST*100, "%")

predichas = np.array(predichas)
print(predichas[0:15])
