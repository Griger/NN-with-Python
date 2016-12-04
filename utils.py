import numpy as np
from constants import *

def saveData(alg, predictedTrain, predictedTest, time, archive):
    lbls = np.load("data/trainLbl.npy")
    lblsTest = np.load("data/testLbl.npy")

    testErrors = NTEST - np.sum(predictedTest == lblsTest)
    trainErrors = NTRAIN - np.sum(predictedTrain == lbls)

    f1 = open('results/'+archive, 'w+')
    print("Algorithm:", alg, file=f1)
    print("Training time:", time, file=f1)
    print("Train error rate:", trainErrors/NTRAIN*100, file=f1)
    print("Test error rate:", testErrors/NTEST*100, file=f1)
    print("Predicted test labels:", file=f1)
    print(''.join(str(i) for i in predictedTest), file=f1)


saveData("pep", [1,2,3], [1,2,3], 41192, 'holis')
