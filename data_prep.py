#IMPORT REQUIRED LIBRARIES
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import csv
#debug info OpenCV version
print ("OpenCV version: " + cv2.__version__)

#SPECIFY REQUIRED IMAGE PATH(S)
#Macropod path
kangarooPath = '/home/deeplearning/Desktop/ReproduceMyThesisInPython/Images/Daytime/Kangaroo'
kangarooProcessPath = '/home/deeplearning/Desktop/ReproduceMyThesisInPython/Images/Daytime/KangarooProcessed/'
onlyKangarooImages = []

# Not Macropod path
notKangarooPath = '/home/deeplearning/Desktop/ReproduceMyThesisInPython/Images/Daytime/NotKangaroo'
onlyNotKangarooImages = []
win_size = (256, 256)

#SPECIFY AND CHECK VALID IMAGE EXTENSIONS
valid_image_extensions = [".png", ".jpeg", ".jpg", ".tif", ".tiff"] #Enter all your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

#SPECIFY PREP-PROCESSING PARAMETERS

#SPECIFY FINAL OUTPUT VARIABLE
imagePaths =[kangarooPath,notKangarooPath]
def data_prep(imagePaths, kangarooProcessPath):  #takes list of paths, and final destination path
    #check if a file is a valid image extensition
    actualImages = []
    finalProcessImageList = []
    imageLabels = []
    for eachPath in imagePaths:
        for file in os.listdir(eachPath):
            extension = os.path.splitext(file)[1]
            if extension.lower() not in valid_image_extensions:
                continue
            actualImages.append(os.path.join(eachPath, file))
            #add image label to the list
            imageLabels.append(os.path.basename(os.path.normpath(eachPath)))  # this assumes that each image path is a class
    n = 1
    #process images
    for imagePath in actualImages:
        image = cv2.imread(imagePath)
        if image is not None:
            #Resize the image
            ResizedImg = cv2.resize(image, win_size)
            #show resized image
            print(imagePath)
            # cv2.imshow("Image", ResizedImg)
            # cv2.waitKey(50)
            #Convert to grayscale
            grayScaleImg = cv2.cvtColor(ResizedImg, cv2.COLOR_RGB2GRAY)
            # show grayScaled image
            #cv2.imshow("Image", grayScaleImg)
            # Perform histogram equalization
            histEqualImg = cv2.equalizeHist(grayScaleImg)
            grayScaleAndEqualImg = np.hstack((grayScaleImg, histEqualImg))  # stacking images side-by-side
            # cv2.imshow("Image", grayScaleAndEqualImg)
            # cv2.waitKey(50)

            plt.subplot(221), plt.imshow(image, 'gray')
            plt.subplot(222), plt.hist(image.flatten(), 256, [0, 256], color='r')
            plt.subplot(223), plt.imshow(histEqualImg, 'gray')
            plt.subplot(224), plt.hist(histEqualImg.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
           # plt.legend('histogram', loc='upper left')
            plt.show(block=False)
            time.sleep(1)
            plt.close()
            #write the image processed folder
            cv2.imwrite(str(kangarooProcessPath) +  'image'+str(n)+'.png', histEqualImg)
            #add the image to the processed image path list
            finalProcessImageList.append(os.path.join(kangarooProcessPath, file))
            #Check if image was added to a correct path
            print(finalProcessImageList[n-1])
            #cv2.waitKey(50)
            n = n + 1
        elif image is None:
            print("Error loading: " + imagePath)
        #cv2.destroyAllWindows()
    #write the class labels to the csv file
    rows = zip(imageLabels)
    outfile = open('./classLabels.csv', 'w')
    writer = csv.writer(outfile)
    writer.writerow(["ClassLabel"])
    for row in rows:
        writer.writerow(row)
#test the function
data_prep(imagePaths, kangarooProcessPath)
# close any open windows
cv2.destroyAllWindows()