import numpy as np
import time
import theano.tensor as T
from theano import function, config

imgs = np.load("data/trainImg.npy")
lbls = np.load("data/trainLbl.npy")

print("Carga completada")
rows = 28
cols = 28
nImgs = len(lbls)

print("Vamos a trabajar con:", nImgs, " imagenes.")


#Initialize random layer weights
w = np.random.rand(10,rows*cols)
b = np.random.rand(10)

print("Estamos trabajando con un array de tipo: ", type(w[0,0]))

a = np.zeros(10, dtype = config.floatX)

print("El tipo que tiene Theano por defecto es:", type(a[0]))

print("¿Son iguales las dimensiones?: ", len(w[0]) == len(imgs[0]))

v = T.vector('v')
w = T.vector('w')

dot = T.dot(v,w)

f = function([v,w], dot)

t0 = time.time()
for i in range(10):
    for img in imgs:
        f(img, img)
t1 = time.time()

print("Hemos tardado %f segundos" % (t1-t0))
