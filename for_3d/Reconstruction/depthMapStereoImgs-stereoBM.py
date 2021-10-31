import numpy as np
import cv2
from matplotlib import pyplot as plt

capL = cv2.VideoCapture(0)
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

        stereo = cv2.StereoBM_create()
        disparity = stereo.compute(imgL, imgR)
        
        cv2.imshow('gray', disparity)
        if cv2.waitKey(1) == ord('q'):
            break

capL.release()
capR.release()
cv2.destroyAllWindows()