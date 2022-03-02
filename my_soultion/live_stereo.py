import cv2

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(2)

while True:
    retL = cam1.read()[1]
    retR = cam2.read()[1]
    cv2.imshow("Left", retL)
    cv2.imshow("Right", retR)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cam1.release()
cam2.release()
