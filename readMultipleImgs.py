from os import listdir
from os.path import isfile, join
import numpy
import cv2

mypath='/home/deeplearning/Desktop/ReproduceMyThesisInPython/Images'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
  print(onlyfiles[n])
print(onlyfiles[0])