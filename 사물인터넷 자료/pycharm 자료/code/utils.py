import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Convolution2D,Flatten,Dense,MaxPooling2D,Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

import random


#### STEP 1 - INITIALIZE DATA
def getName(filePath):
    myImagePathL = filePath.split('/')[-2:]
    myImagePath = os.path.join(myImagePathL[0],myImagePathL[1])
    return myImagePath

def importDataInfo(path):
    columns = ['Center','Steering', 'Throttle']
    noOfFolders = len(os.listdir(path))//2
    data = pd.DataFrame()
    for x in range(0,26):
        dataNew = pd.read_csv(os.path.join(path, f'log_{x}.csv'), names = columns)
        print(f'{x}:{dataNew.shape[0]} ',end='')
        #### REMOVE FILE PATH AND GET ONLY FILE NAME
        #print(getName(data['center'][0]))
        dataNew['Center']=dataNew['Center'].apply(getName)
        data =data.append(dataNew,True )
    print(' ')
    print('Total Images Imported',data.shape[0])
    return data

#### STEP 2 - VISUALIZE AND BALANCE DATA
def balanceData(data,display=True):
    nBin = 31
    nBin1 = 3
    samplesPerBin =  5000
    hist, bins = np.histogram(data['Steering'], nBin)
    hist1, bins1 = np.histogram(data['Throttle'], nBin1)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        center1 = (bins1[:-1] + bins1[1:]) * 0.5
        plt.figure(1)
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
        plt.figure(2)
        plt.hist(data['Throttle'])
        plt.plot((np.min(data['Throttle']), np.max(data['Throttle'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('Throttle Value')
        plt.ylabel('No of Samples')
        plt.show()

    removeindexList = []
    removeindexList_Throttle = []
    print("Total Throttle(before removal) : ", set(data['Throttle']))
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if (data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]):
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    # Throttle Removal


    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    data.reset_index(drop=False, inplace=True)
    print(data)


    for i in range(len(data['Steering'])):
        if data['Throttle'][i]<0.3:
            removeindexList_Throttle.append(i)
    print('Removed Throttle value', len(removeindexList_Throttle))

    data.drop(data.index[removeindexList_Throttle], inplace=True)
    print('Remaining Images:', len(data))
    print("Min Throttle : ", np.min(data['Throttle']))
    print("Max Throttle : ", np.max(data['Throttle']))
    print("Total Throttle : ", set(data['Throttle']))

    if display:
        hist, _ = np.histogram(data['Steering'], (nBin))

        plt.figure(1)
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('Angle')
        plt.ylabel('No of Samples')
        plt.show()
        plt.figure(2)
        plt.hist(data['Throttle'])
        #plt.plot((np.min(data['Throttle']), np.max(data['Throttle'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('Throttle Value')
        plt.ylabel('No of Samples')
        plt.show()
    return data

#### STEP 3 - PREPARE FOR PROCESSING
def loadData(path, data):
  imagesPath = []
  Throttle =[]
  Steering = []

  for i in range(len(data)):
    indexed_data = data.iloc[i]
    imagesPath.append( os.path.join(path,indexed_data[1]))
    Steering.append(float(indexed_data[2]))
    Throttle.append(float(indexed_data[3]))


  Total_set = np.vstack((Steering, Throttle))
  imagesPath = np.asarray(imagesPath)
  Total_set_T = np.transpose(Total_set)
  y_label = np.asarray(Total_set_T)
  print(y_label)
  return imagesPath, y_label


#### STEP 5 - AUGMENT DATA
def augmentImage(imgPath):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    return img


#### STEP 6 - PREPROCESS
def preProcess(img):
    img = img[:,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (11, 11), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


#### STEP 7 - CREATE MODEL
def createModel():
  model = Sequential()

  model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
  model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
  model.add(Dropout(rate=0.2))
  model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
  model.add(Convolution2D(64, (3, 3), activation='elu'))
  model.add(Convolution2D(64, (3, 3), activation='elu'))
  model.add(Flatten())
  model.add(Dense(100, activation = 'elu'))
  model.add(Dropout(rate=0.3))
  model.add(Dense(50, activation = 'elu'))
  model.add(Dense(10, activation = 'elu'))
  model.add(Dense(2))
  model.summary()
  model.compile(Adam(lr=0.001),loss='mse')

  return model

#### STEP 8 - TRAINNING
def dataGen(imagesPath, STList, batchSize, trainFlag):
    while True:
        imgBatch = []
        STBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img = augmentImage(imagesPath[index])
            else:
                img = mpimg.imread(imagesPath[index])
            steering_throttle = STList[index]
            img = preProcess(img)
            imgBatch.append(img)
            STBatch.append(steering_throttle)

        yield (np.asarray(imgBatch),np.asarray(STBatch))