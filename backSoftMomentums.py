import numpy as np
import time
import theano.tensor as T
from theano import function, config, scan, shared
from constants import *
from utils import saveData

np.random.seed(12345678)

lr = 0.1
eta = 0.1
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

prevDeltaW1 = shared(np.zeros((784, nHidden), dtype = config.floatX), name = 'prevDeltaW1')
prevDeltab1 = shared(np.zeros(nHidden, dtype = config.floatX), name = 'prevDeltab1')
prevDeltaW2 = shared(np.zeros((nHidden, 10), dtype = config.floatX), name = 'prevDeltaW2')
prevDeltab2 = shared(np.zeros(10, dtype = config.floatX), name = 'prevDeltab2')

#sigmoid like activation function
hid = 1.0 / (1.0 + T.exp(-(T.dot(x, W1) + b1)))
out = 1.0 / (1.0 + T.exp(-(T.dot(hid, W2) + b2)))

y_hat = T.nnet.softmax(out)
#err = 0.5 * T.sum(y - y_hat) ** 2 #mean square error
err = - T.sum(y * T.log(y_hat)) #cross-entropy error
prediction = T.argmax(y_hat)

#define gradients
dW1, db1, dW2, db2 = T.grad(err, [W1, b1, W2, b2])

predict = function([x], prediction)

##NN training
train = function([x, y], err,
    updates={
        (W2, W2 - lr * dW2 - eta * prevDeltaW2),
        (b1, b1 - lr * db1 - eta * prevDeltab1),
        (W1, W1 - lr * dW1 - eta * prevDeltaW1),
        (b2, b2 - lr * db2 - eta * prevDeltab2),
        (prevDeltab2, db2),
        (prevDeltaW1, dW1),
        (prevDeltaW2, dW2),
        (prevDeltab1, db1)})

#functions to make a bunch of predictions from an image matrix 
A = T.matrix('A')
hidMatrix = 1.0 / (1.0 + T.exp(-(T.dot(A, W1) + b1)))
outMatrix = 1.0 / (1.0 + T.exp(-(T.dot(hidMatrix, W2) + b2)))
y_hatMatrix = T.nnet.softmax(outMatrix)
predictions = T.argmax(y_hatMatrix, axis = 1)
predictMatrix = function([A], predictions)

# train the NN
t0 = time.time()
for i in range(epochs):
    print("Epoch",i)
    for img, lbl, idx in zip(imgs, binLbls, range(NTRAIN)):
        train(img, lbl)
t1 = time.time()

predictedClasses = predictMatrix(imgsTest)
nErrorTest = NTEST - np.sum(predictedClasses == lblsTest)
print("Test errors:", nErrorTest, "%:", nErrorTest/NTEST*100.0)

print("Training time:", (t1-t0))
fileName = "backSoftwithMomentumsLR"+str(lr)+"EPOCH"+str(epochs)+"NHID"+str(nHidden)+"ETA"+str(eta)+".npz"
np.savez("weights/"+fileName, W1 = W1.get_value(), b1 = b1.get_value(), W2 = W2.get_value(), b2 = b2.get_value())

nErrorTrain = NTRAIN - np.sum(predictMatrix(imgs) == lbls)
print("Train errors:", nErrorTrain, "%:", nErrorTrain/NTRAIN*100)

algDescription = "Algoritmo con backpropagation en una NN con una capa oculta de \n"
algDescription += str(nHidden) + " neuronas con función de activación sigmoidal. Y una capa de\n"
algDescription += "salida tipo softmax. Usando una tasa de aprenzidaje de" + str(lr) +"\n y dando "
algDescription += str(epochs) + " al conjunto de train, usando además momentos con una tasa de" + str(eta)
saveData(algDescription, predictMatrix(imgs), predictMatrix(imgsTest), t1-t0, "b1withMomentumsLR"+str(lr)+"EPOCH"+str(epochs)+"NHID"+str(nHidden)+"ETA"+str(eta)+".txt")
