import cv2

cap = cv2.VideoCapture(1)	# Left Cam
ret = cap.read()[1]
imgL = cv2.imwrite('imgL.jpg', ret)

cap = cv2.VideoCapture(2)	# Right Cam
ret = cap.read()[1]
imgR = cv2.imwrite('imgR.jpg', ret)