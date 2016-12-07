import numpy as np
import time
import theano.tensor as T
from theano import function, config, scan, shared
from constants import *
from utils import saveData

np.random.seed(12345678)

lr = 0.1
nHidden = 256
epochs = 100

imgs = np.load("data/trainImg.npy")
lbls = np.load("data/trainLbl.npy")
binLbls = np.load("data/binTrainLbl.npy")
imgsTest = np.load("data/testImg.npy")
lblsTest = np.load("data/testLbl.npy")

x = T.vector('x')
y = T.vector('y')
W1 = shared(np.random.normal(loc=0, scale=.1, size=(784, nHidden)).astype(config.floatX), name = 'W1')
b1 = shared(np.random.normal(loc=0, scale=.1, size=nHidden).astype(config.floatX), name = 'b1')
W2 = shared(np.random.normal(loc=0, scale=.1, size=(nHidden, 10)).astype(config.floatX), name = 'W2')
b2 = shared(np.random.normal(loc=0, scale=.1, size=10).astype(config.floatX), name = 'b2')

#tanh like activation function
# hid = T.tanh(T.dot(x, W1) + b1)
# out = T.tanh(T.dot(hid, W2) + b2)

#sigmoid like activation function
hid = 1.0 / (1.0 + T.exp(-(T.dot(x, W1) + b1)))
out = 1.0 / (1.0 + T.exp(-(T.dot(hid, W2) + b2)))

#Lecun recommended tanh like activation function
# hid = 1.7159 * T.tanh(0.67 * (T.dot(x, W1) + b1) )
# out = 1.7159 * T.tanh(0.67 * (T.dot(hid, W2) + b2) )

y_hat = T.nnet.softmax(out)
#err = 0.5 * T.sum(y - y_hat) ** 2 #mean square error
err = - T.sum(y * T.log(y_hat)) #cross-entropy error
prediction = T.argmax(y_hat)

#define gradients
dW1, db1, dW2, db2 = T.grad(err, [W1, b1, W2, b2])

salida = function([x], y_hat)
predict = function([x], prediction)
train = function([x, y], err,
    updates={
        (W2, W2 - lr * dW2),
        (b1, b1 - lr * db1),
        (W1, W1 - lr * dW1),
        (b2, b2 - lr * db2)})

A = T.matrix('A')
hidMatrix = 1.0 / (1.0 + T.exp(-(T.dot(A, W1) + b1)))
outMatrix = 1.0 / (1.0 + T.exp(-(T.dot(hidMatrix, W2) + b2)))
y_hatMatrix = T.nnet.softmax(outMatrix)
predictions = T.argmax(y_hatMatrix, axis = 1)
predictMatrix = function([A], predictions)



# now do the computations
t0 = time.time()
cadenaTest = "c("
for i in range(epochs):
    print("Epoch",i)
    for img, lbl, idx in zip(imgs, binLbls, range(NTRAIN)):
        train(img, lbl)
    if (i % 5 == 0):
        print("Ya van:", i, " epochs.")
        predictedClasses = predictMatrix(imgsTest)
        nErrorTest = NTEST - np.sum(predictedClasses == lblsTest)
        print("Test errors:", nErrorTest, "%:", nErrorTest/NTEST*100.0)
        cadenaTest += str(nErrorTest/NTEST*100.0) + ","
        np.savez("weights/"+"backSoftwithLecunActFunLR"+str(lr)+"EPOCH"+str(i)+"NHID"+str(nHidden)+".npz", W1 = W1.get_value(), b1 = b1.get_value(), W2 = W2.get_value(), b2 = b2.get_value())
t1 = time.time()

"""
nErrorTest = 0
predictedTest = []
for img, lbl in zip(imgsTest, lblsTest):
    p = predict(img)
    predictedTest.append(p)
    if p != lbl:
        nErrorTest += 1

print("Test errors:", nErrorTest, "%:", nErrorTest/NTEST*100.0)
"""

predictedClasses = predictMatrix(imgsTest)
nErrorTest = NTEST - np.sum(predictedClasses == lblsTest)
print("Test errors:", nErrorTest, "%:", nErrorTest/NTEST*100.0)

cadenaTest += str(nErrorTest/NTEST*100.0) + ")"
print(cadenaTest)
print("Training time:", (t1-t0))
np.savez("weights/"+"backSoft.npz", W1 = W1.get_value(), b1 = b1.get_value(), W2 = W2.get_value(), b2 = b2.get_value())

"""
t0 = time.time()
predictedTrain = []
nErrorTrain = 0
for img, lbl in zip(imgs, lbls):
    p = predict(img)
    predictedTrain.append(p)
    if p != lbl:
        nErrorTrain += 1
t1 = time.time()

print("Hemos tardado en calcular el error con un for", (t1-t0))
print("Train errors:", nErrorTrain, "%:", nErrorTrain/NTRAIN*100)
"""

nErrorTrain = NTRAIN - np.sum(predictMatrix(imgs) == lbls)
print("Train errors:", nErrorTrain, "%:", nErrorTrain/NTRAIN*100)

algDescription = "Algoritmo con backpropagation en una NN con una capa oculta de \n"
algDescription += str(nHidden) + " neuronas con función de activación la tanh de Lecun. Y una capa de\n"
algDescription += "salida tipo softmax. Usando una tasa de aprenzidaje de" + str(lr) +"\n y dando "
algDescription += str(epochs) + " al conjunto de train."
saveData(algDescription, predictMatrix(imgs), predictMatrix(imgsTest), t1-t0, "b1withLeCunActFun.txt")
