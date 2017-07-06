import cv2
from os import listdir
from os.path import isfile, join
import numpy
from sklearn.neighbors import KNeighborsClassifier
# simple HOG feature extraction using 64 x 128 image
# setup global variables
finalFeatures = []

# 1 Step 1 is to decide the size of image patch - for simplicity and faster processing we choose
# 64x128
win_size = (256, 256)
img = cv2.imread("dayImg1.jpg")
img = cv2.resize(img, win_size)
# Add images to the list


# 2 Step2 organize the list of images and perform preprocessing steps in this simple example, we just convert RGB to grayscale
# list of images
mypath = '/home/deeplearning/Desktop/ReproduceMyThesisInPython/Images'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))
    print(onlyfiles[n])
    # resize images
    images[n] = cv2.resize(images[n], win_size)
    images[n] = cv2.cvtColor(images[n], cv2.COLOR_RGB2GRAY)
    cv2.imshow('image', images[n])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 3 Configure HOG parameters
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# 4 Extract HOG feature points and show the points
d = cv2.HOGDescriptor(win_size, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                      histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
# 5 Compute HOG Descriptor for each image in the list
print(images[1].shape)  # just a check to see if the list was populated properly
for eachImg in images:
    hog = d.compute(eachImg)
    print(hog.shape)
    # print(hog[0])
    # print(type(hog))
    compactness, labels, hog = cv2.kmeans(hog, 5376, None, criteria, 10, flags)
    finalFeatures.append(hog)
print(finalFeatures[0][0])
numpyMatrix = numpy.array([numpy.array(x) for x in finalFeatures])
print(numpyMatrix)
print(type(numpyMatrix))
print(numpyMatrix.shape)
print(finalFeatures.shape)
y = [0, 1, 1, 0]
neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(finalFeatures, y)
