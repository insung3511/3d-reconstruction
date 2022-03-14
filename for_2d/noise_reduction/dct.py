from scipy.fftpack import dct
import numpy as np
import cv2

x = np.array([1, 2, 3, 4], dtype=np.float32)
a = cv2.dct(x)
print(a)

b = dct(x, axis=0, norm='ortho', type=2)
print(b)

x = np.random.random((4,4,3))
a = [cv2.dct(x[:, :, i]) for i in range(3)]
b = dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")
# a == b
import time 
y = np.random.random((5000, 6, 6, 3))
t1 = time.time()
a = np.zeros((5000, 6, 6, 3))
for i in range(5000):
    for j in range(3):
        a[i, :, :, j] = cv2.dct(y[i, :, :, j])
print("cost time 1: ", time.time()-t1)
# cost time 1:  0.05182027816772461
t2 = time.time()
b = dct(dct(y, axis=1, norm="ortho"), axis=2, norm="ortho")
print("cost time 2: ", time.time()-t2)
# cost time 2:  0.014388084411621094

x = np.random.random((4,4,3))
a = [cv2.dct(x[:, :, i]) for i in range(3)]
b = dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")
# a == b
import time 
y = np.random.random((5000, 6, 6, 3))
t1 = time.time()
a = np.zeros((5000, 6, 6, 3))
for i in range(5000):
    for j in range(3):
        a[i, :, :, j] = cv2.dct(y[i, :, :, j])
print("cost time 1: ", time.time()-t1)
# cost time 1:  0.05182027816772461
t2 = time.time()
b = dct(dct(y, axis=1, norm="ortho"), axis=2, norm="ortho")
print("cost time 2: ", time.time()-t2)
# cost time 2:  0.014388084411621094