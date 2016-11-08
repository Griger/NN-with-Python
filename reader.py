import gzip
import struct
import numpy as np

def readInt (file):
    return struct.unpack('>i', file.read(4))[0]

def readUnsignedByte (file):
    return struct.unpack('B', file.read(1))[0]

def readImages (filename):
    file = gzip.GzipFile(filename, 'rb')
    magicNumber = readInt(file)
    nImages = readInt(file)
    rows = readInt(file)
    cols = readInt(file)

    print("Nº images: " + str(nImages) + " Rows: " + str(rows) + " Cols: " + str(cols))

    imageLimit = 100
    images = np.zeros((imageLimit, rows, cols))

    for i in range(imageLimit):
        for j in range(rows):
            for k in range(cols):
                images[i,j,k] = readUnsignedByte(file)

    return images

def normalize (image):
    rows = 28
    cols = 28
    normalizedIm = np.zeros((rows,cols))

    for i in range(rows):
        for j in range(cols):
            normalizedIm[i,j] = image[i,j]*1.0/255.0

    return normalizedIm


images = readImages("mnist.data/"+"t10k-images-idx3-ubyte.gz")

normalizedIm = normalize(images[61])
print("Normalized images: ")
print(normalizedIm)

for i in range(28):
    for j in range(28):
        if (normalizedIm[i,j] != 0.0):
            print(1, end="")
        else:
            print(0, end="")
    print("")
