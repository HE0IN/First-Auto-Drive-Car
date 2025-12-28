import cv2
import numpy as np
import os
import tensorflow
from tensorflow.keras.models import load_model
import WebcamModule as wM
import MotorModule as mM


#######################################
throttleSen = 0.575
steeringSen = 0.175  # Steering Sensitivity
motor= mM.Motor(0,6,5,26,19,13)#Pin Numbers
#print("finish")
model = load_model('/home/pi/example/ai/suc_9.h5')

######################################

def preProcess(img):
    img = img[:, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

while True:
    img = wM.getImg(True, size=[240, 120])
    img = np.asarray(img)
    img = preProcess(img)
    img = np.array([img])
    pred = model.predict(img)
    print(pred)
    motor.move(-pred[0][1]*throttleSen, pred[0][0]*steeringSen)
    cv2.waitKey(1)