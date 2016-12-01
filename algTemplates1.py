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
b = shared(np.zeros(10, dtype = config.floatX))

print(w[0])
w[0] += 1
print(w[0])

v = T.vector('v')
u = T.vector('u')
X = T.matrix('X')

f = function([u,X], T.dot(u,X))

t0 = time.time()
for img, lbl in zip(imgs, lbls):
    predicha = np.argmax(f(img, w))
    #print("Clase predicha:", predicha, "Clase real:", lbl)
    if (predicha == lbl):
        #print("Acierto")
        w[img > 0,predicha] += lr
    else:
        #print("Error")
        w[img > 0, int(lbl)] += lr
        w[img > 0, predicha] -= lr
t1 = time.time()

print("Hemos tardado: ", (t1-t0))

errores = 0

for img, lbl in zip(imgsTest, lblsTest):
    predicha = np.argmax(f(img, w))
    if (predicha != lbl):
        errores += 1

print("Se han obtenido", errores, "errores. Un", errores/NTEST*100, "%")

"""
for img in imgs[0:20]:
    print(f(w[0], img))


sx = T.vector()
sy = T.matrix()
prediction = T.argmax(T.dot(sx, w))
l = T.scalar()

algTemplates1 = function([sx, l], prediction,
    updates{
        w[prediction] += lr*(#+ o - segun se equivoque o no)
        b[prediction] += lr*(#+ o - segun se equivoque o no)
    })

for img, lbl in imgs, lbls:
    algTemplates1(img, lbl)
"""

#Algoritmo de prediccion
"""
predictTemplate = function([sx], prediccion)

for img in testImgs:
    np.append(predictTemplate(img))
"""
# y sacamos el error
