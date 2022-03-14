from scipy.fftpack import dct
import numpy as np
import cv2

x = np.array([1, 2, 3, 4], dtype=np.float32)
a = cv2.dct(x)
print(a)

b = dct()