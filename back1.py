import numpy as np
import time
import theano.tensor as T
from theano import function, config, scan, shared
from constants import *

np.random.seed(12345678)

lr = 0.1

imgs = np.load("data/trainImg.npy")
lbls = np.load("data/trainLbl.npy")
binLbls = np.load("data/binTrainLbl.npy")
imgsTest = np.load("data/testImg.npy")
lblsTest = np.load("data/testLbl.npy")

# symbol declarations
x = T.vector('x')
y = T.vector('y')
w = shared(np.random.normal(loc=0, scale=.1, size=(784, 256)).astype(config.floatX))
b = shared(np.zeros(256, dtype = config.floatX))
v = shared(np.zeros((256, 10), dtype = config.floatX))
c = shared(np.zeros(10, dtype = config.floatX))

# symbolic expression-building
hid = T.tanh(T.dot(x, w) + b)
out = T.tanh(T.dot(hid, v) + c)
err = T.sum(out - y)
gw, gb, gv, gc = T.grad(err, [w, b, v, c])

# compile a fast training function
train = function([x, y], err,
    updates={
        w: w - lr * gw,
        b: b - lr * gb,
        v: v - lr * gv,
        c: c - lr * gc})

print(w.get_value())
print(b.get_value())
print(v.get_value())
print(c.get_value())

train(imgs[0], binLbls[0])

print("w"w.get_value())
print(b.get_value())
print(v.get_value())
print(c.get_value())

"""
# now do the computations
t0 = time.time()
for img, lbl, idx in zip(imgs, binLbls, range(NTRAIN)):
    train(img, lbl)
    if (idx % 10000 == 0):
        print("Entrenado con", idx, "imagenes")
t1 = time.time()

salida = T.argmax(out)
clasificar = function([x], salida)

print(w.get_value())
print(b.get_value())
print(v.get_value())
print(c.get_value())

nErrorTest = 0
for img, lbl in zip(imgsTest, lblsTest):
    if clasificar(img) != lbl:
        nErrorTest += 1


nErrorTrain = 0
for img, lbl in zip(imgs, lbls):
    if clasificar(img) != lbl:
        nErrorTrain += 1


print("Tiempo de entrenamiento:", t1-t0)
#print("Errores train:", nErrorTrain, "%:", nErrorTrain/NTRAIN*100)
print("Errores test:", nErrorTest, "%:", nErrorTest/NTEST*100.0)
"""
