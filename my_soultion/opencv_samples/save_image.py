import cv2

def save_image():
    cap = cv2.VideoCapture(0)	# Left Cam
    ret = cap.read()[1]
    #imgL = cv2.imwrite('imgL.jpg', cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY))
    imgL = cv2.imwrite('imgL.jpg', ret)
    print("Take Picture [LEFT]")

    cap = cv2.VideoCapture(2)	# Right Cam
    ret = cap.read()[1]
    #imgR = cv2.imwrite('imgR.jpg', cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY))
    imgR = cv2.imwrite('imgR.jpg', ret)
    print("Take Picture [RIGHT]")
    print("### ALL DONE ###")