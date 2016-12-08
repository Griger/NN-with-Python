import numpy as np
import time
import theano.tensor as T
import theano.tensor.signal.pool as pool
from theano import function, config, scan, shared
from constants import *
from utils import saveData

class convPoolLayer:
    def __init__(self, input, filter_shape, image_shape):
        self.input = input

        self.W = shared(np.random.normal(loc=0, scale=.1, size=filter_shape).astype(config.floatX), borrow = True)
        self.b = shared(np.random.normal(loc=0, scale=.1, size=filter_shape[0]).astype(config.floatX), borrow = True)

        conv_out = T.nnet.conv2d(input = input, filters = self.W, filter_shape = filter_shape, input_shape = image_shape)
        pooled_out = pool.pool_2d(input = conv_out, ds = (2,2), ignore_border  = True)

        self.output = 1.0 / (1.0 + T.exp(-(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))))
        self.params = [self.W, self.b]
        self.input = input

class hiddenLayer:
    def __init__(self, input, dimIn, nHidden):
        self.input = input

        self.W = shared(np.random.normal(loc=0, scale=.1, size=(dimIn, nHidden)).astype(config.floatX), borrow = True)
        self.b = shared(np.random.normal(loc=0, scale=.1, size=nHidden).astype(config.floatX), borrow = True)

        self.output = 1.0 / (1.0 + T.exp(-(T.dot(input, self.W) + self.b)))
        self.params = [self.W, self.b]

class outputLayer:
    def __init__(self, input, dimIn):
        self.input = input

        self.W = shared(np.random.normal(loc=0, scale=.1, size=(dimIn, 10)).astype(config.floatX), borrow = True)
        self.b = shared(np.random.normal(loc=0, scale=.1, size=10).astype(config.floatX), borrow = True)

        output = 1.0 / (1.0 + T.exp(-(T.dot(input, self.W) + self.b)))
        y_hat = T.nnet.softmax(output)
        self.output = y_hat
        self.params = [self.W, self.b]

#main code
np.random.seed(12345678)

imgs = shared(np.load("data/trainMtxImg.npy"), borrow = True)
lbls = shared(np.load("data/trainLbl.npy"), borrow = True)
binLbls = shared(np.load("data/binTrainLbl.npy"), borrow = True)
imgsTest = shared(np.load("data/testMtxImg.npy"), borrow = True)
lblsTest = shared(np.load("data/testLbl.npy"), borrow = True)

index = T.lscalar()
x = T.matrix('x')
y = T.vector('y')

#We're going to use an on-line learning method
image_shape = (1,1,COLS, ROWS)
layer0_input = x.reshape(image_shape)

#Number of convolutional filters per convolutional layer
nFilters = [6, 256]

layer0 = convPoolLayer(input = layer0_input, filter_shape = (nFilters[0], 1, 5, 5), image_shape = image_shape)
layer1 = convPoolLayer(input = layer0.output, filter_shape = (nFilters[1], nFilters[0], 5, 5), image_shape = (1, nFilters[0], 12, 12))

layer2_input = layer1.output.flatten(2)

nHidden = 256
layer2 = hiddenLayer(input = layer2_input, dimIn = nFilters[1]*4*4, nHidden = nHidden)

layer3 = outputLayer(input = layer2.output, dimIn = nHidden)
err = - T.sum(y * T.log(layer3.output))
prediction = T.argmax(layer3.output)

params = layer3.params + layer2.params + layer1.params + layer0.params
grads = T.grad(err, params)

lr = 0.1
epochs = 41

updates = [(param_i, param_i - lr * grad_i) for param_i, grad_i in zip(params, grads)]

train = function([index], err, updates = updates,
                givens = {
                    x: imgs[index],
                    y: binLbls[index]
                })

predict = function([x], prediction)

t0 = time.time()
for i in range(epochs):
    print("Epoch",i)

    for j in range(NTRAIN):
        train(j)
    # print("Tiempo invertido en una epoch:", (t1-t0))
    # if (i % 5 == 0):
    #     errors = 0
    #     print("Ya van", i, "epochs.")
    #     for img, lbl in zip(imgsTest.get_value(), lblsTest.get_value()):
    #         if (predict(img) != lbl):
    #             errors += 1
    #     print("Errores:", errors, "Tasa:", errors*100.0/NTEST)
t1 = time.time()

print("Training time", (t1-t0))
filename = "basicconvolutional256"
np.savez("weights/"+filename, W0 = layer0.W.get_value(), b0 = layer0.b.get_value(), W1 = layer1.W.get_value(), b1 = layer1.b.get_value(), W2 = layer2.W.get_value(), b2 = layer2.b.get_value(), W3 = layer3.W.get_value(), b3 = layer3.b.get_value())

errors = 0
predictedTest = []
for img, lbl in zip(imgsTest.get_value(), lblsTest.get_value()):
        p = predict(img)
        predictedTest.append(p)
        if (p != lbl):
            errors += 1

print("Test errors:", errors, "Error rate:", errors*100.0/NTEST)

errors = 0
predictedTrain = []
for img, lbl in zip(imgs.get_value(), lbls.get_value()):
        p = predict(img)
        predictedTrain.append(p)
        if (p != lbl):
            errors += 1

print("Train errors:", errors, "Error rate:", errors*100.0/NTRAIN)

algDescription = "Algoritmo con backpropagation en una CNN con una capa oculta de \n"
algDescription += str(nHidden) + " neuronas con función de activación sigmoidal. Y una capa de\n"
algDescription += "salida tipo softmax. Usando una tasa de aprenzidaje de" + str(lr) +"\n y dando "
algDescription += str(epochs) + " al conjunto de train. Previamente se pasand dos filtros de convolucion"
algDescription += " y pooling (/2). Ambos filtros son 5x5, en la primera capa hay 6 y en el otro 256."
saveData(algDescription, predictedTrain, predictedTest, t1-t0, filename+".txt")
