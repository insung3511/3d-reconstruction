import cv2

def save_image():
    cap = cv2.VideoCapture(0)	# Left Cam
    ret = cap.read()[1]
    imgL = cv2.imwrite('imgL.jpg', ret)
    
    cap = cv2.VideoCapture(2)	# Right Cam
    ret = cap.read()[1]
    imgR = cv2.imwrite('imgR.jpg', ret)
