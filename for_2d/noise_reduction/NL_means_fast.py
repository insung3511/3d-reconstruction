from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread('./data/testing2.jpg')

dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
 
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()