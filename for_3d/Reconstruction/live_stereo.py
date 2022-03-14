import cv2

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)
cam3 = cv2.VideoCapture(2)

while True:
    testcam_1 = cam1.read()[1]
    testcam_2 = cam2.read()[1]
    testcam_3 = cam3.read()[1]
    
    #cv2.imshow("Left", retL)
    #cv2.imshow("Right", retR)

    cv2.imshow("Test Cam1", testcam_1)
    cv2.imshow("Test Cam2", testcam_2)
    cv2.imshow("Test Cam3", testcam_3)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cam1.release()
cam2.release()
