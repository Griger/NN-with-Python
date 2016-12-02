import gzip
import struct
import numpy as np
from theano import config

def readInt (file):
    """Function that reads an integer from a file with big-endian order

    :param file: file from we read
    :returns: read int
    """
    return struct.unpack('>i', file.read(4))[0]

def readUnsignedByte (file):
    """Function that reads an unsigned char for a file with native order

    :param file: file from we read
    :returns: read unsigned char
    """
    return struct.unpack('B', file.read(1))[0]

def normalizeImage (image):
    normalizedIm = image*1.0/255.0
    return normalizedIm

# Function that loads MNIST images from a file
def readImages (filename):
    file = gzip.GzipFile(filename, 'rb')
    magicNumber = readInt(file)
    nImages = readInt(file)
    rows = readInt(file)
    cols = readInt(file)

    imageLimit = nImages
    images = np.zeros((imageLimit, rows*cols), dtype = config.floatX)

    #Load images
    for i in range(imageLimit):
        for j in range(rows*cols):
            images[i,j] = readUnsignedByte(file)

    #Normalize images
    normImages = np.apply_along_axis(normalizeImage, 0, images)

    return normImages

def readLabels (filename):
    file = gzip.GzipFile(filename, 'rb')
    magicNumber = readInt(file)
    size = readInt(file)

    labels = np.zeros(size, dtype = config.floatX)

    for i in range(size):
        labels[i] = readUnsignedByte(file)

    return labels

def labelsToBinary (lbls):
    binaryLbls = np.zeros((len(lbls), 10), dtype = config.floatX)

    for lbl, binarylbl in zip(lbls, binaryLbls):
        binarylbl[int(lbl)] = 1.0

    return binaryLbls

def printFormattedImage (image):
    for i in range(28):
        for j in range(28):
            if (image[i*28+j] != 0.0):
                print(1, end="")
            else:
                print(0, end="")
        print("")

#Load images and labels from DB files
images = readImages("mnist.data/"+"train-images-idx3-ubyte.gz")
labels = readLabels("mnist.data/"+"train-labels-idx1-ubyte.gz")
binLabels = labelsToBinary(labels)
imagesTest = readImages("mnist.data/"+"t10k-images-idx3-ubyte.gz")
labelsTest = readLabels("mnist.data/"+"t10k-labels-idx1-ubyte.gz")

#Save loaded images and labels
np.save("data/"+"trainImg.npy", images)
np.save("data/"+"trainLbl.npy", labels)
np.save("data/"+"testImg.npy", imagesTest)
np.save("data/"+"testLbl.npy", labelsTest)
np.save("data/"+"binTrainLbl.npy", binLabels)
