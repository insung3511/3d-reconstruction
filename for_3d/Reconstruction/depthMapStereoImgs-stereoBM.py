import numpy as np
import cv2
from matplotlib import pyplot as plt

capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)

while True:
    if capR and capL is False:
        print("Camera NOT DETECTED")
        break
    
    else:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        imgL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        '''imgL = cv2.imread('left01.jpg',cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread('left02.jpg',cv2.IMREAD_GRAYSCALE)
'''
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL,imgR)
        
        cv2.imshow('gray', disparity)

        if cv2.waitKey(1) == ord('q'):
            break

capL.release()
capR.release()
cv2.destoryAllWindows()