import cv2
import numpy as np
import os
import tensorflow
from tensorflow.keras.models import load_model
import WebcamModule_Final as wM
import MotorModule as mM
import time
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)

############### Ultra Sonic ###############
TRIG=23
ECHO=24

GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
#######################################

throttleSen = 0.525
steeringSen = 0.225  # Steering Sensitivity
motor= mM.Motor(0,6,5,26,19,13)#Pin Numbers
model = load_model('/home/pi/example/ai/12210130.h5')

def preProcess(img):
    img = img[:, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

while True:
    GPIO.output(TRIG, False)
    time.sleep(0.0001)

    GPIO.output(TRIG, True)
    time.sleep(0.0001)
    GPIO.output(TRIG, False)
    while GPIO.input(ECHO)==0:
        pulse_start = time.time()

    while GPIO.input(ECHO)==1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration *17150
    distance = round(distance, 2)


    img, StopSign= wM.getImg(True, size=[240, 120])
    img = np.asarray(img)
    img = preProcess(img)
    img = np.array([img])
    pred = model.predict(img)
    print(pred)

    motor.move(-pred[0][1]*throttleSen, pred[0][0]*steeringSen)

    if distance < 30 | int(StopSign) > 1:
        print("Stop")
        motor.move(0,0)
    else:
        print("Go!!")
    cv2.waitKey(1)
