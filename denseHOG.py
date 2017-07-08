import cv2
from os import listdir
from os.path import isfile, join
import numpy
import pandas as pd

# GLOBAL VARIABLES
finalFeatures = []

#IMAGE PATHS
#processedImagesPath = '/home/deeplearning/Desktop/ReproduceMyThesisInPython/Images/Daytime/KangarooProcessed/'
processedImagesPath = '/home/deeplearning/Desktop/ReproduceMyThesisInPython/Images/smallsize/'
onlyfiles = [ f for f in listdir(processedImagesPath) if isfile(join(processedImagesPath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(processedImagesPath,onlyfiles[n]))

# HOG FEATURE PARAMETERS
win_size = (256, 256)
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

# HOG FEATURE EXTRACTION SETUP
d = cv2.HOGDescriptor(win_size, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                      histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
# COMPUTE HOG DESCRIPTOR EACH IMAGE
n=0;
for eachImg in images:
    print('--------------Computing HOG for-------------------')
    print(processedImagesPath+''+onlyfiles[n])
    hog = d.compute(eachImg)
    compactness, labels, hog = cv2.kmeans(hog, 5376, None, criteria, 10, flags) # run k-means algorithms to select important features
    hog = hog.ravel()
    print('--------------Feature vector size is:-------------')
    print(hog.shape)
    finalFeatures.append(hog)
    n = n+1
#print(finalFeatures[0][0])
numpyMatrix = numpy.array([numpy.array(x) for x in finalFeatures])
#Save the features to csv file
numpy.savetxt("hog.csv", numpyMatrix, delimiter=",")



