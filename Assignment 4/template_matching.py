import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

P = 60

def _debug_print(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def exhaustive_search(test, reference):
    refx = reference.shape[0]
    refy = reference.shape[1]
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    ref_padded = np.pad(reference, ((0, test.shape[0] - refx), (0, test.shape[1] - refy)),\
                        'constant', constant_values=((0,0),(0,0)))
    
    c = np.real(np.fft.ifft2((np.conj(np.fft.fft2(ref_padded))*np.fft.fft2(test))/\
                             np.absolute(np.conj(np.fft.fft2(ref_padded))*np.fft.fft2(test))))
    
    temp =  np.unravel_index(np.argmax(c, axis=None), c.shape)
    return int(temp[0] + refx/2), int(temp[1] + refy/2)

def exhaustive_search2(test, reference):
    refx = reference.shape[0]
    refy = reference.shape[1]
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    
    c = np.zeros((test.shape[0] - refx + 1, test.shape[1] - refy + 1))
    for i in range(test.shape[0] - refx + 1):
        for j in range(test.shape[1] - refy + 1):
            c[i,j] = np.sum(reference.astype(int) * test[i:i+refx, j:j+refy].astype(int))/\
                        (np.linalg.norm(reference) * np.linalg.norm(test[i:i+refx, j:j+refy]))
    #print(test, reference)
    temp =  np.unravel_index(np.argmax(c, axis=None), c.shape)
    return int(temp[0] + refx/2), int(temp[1] + refy/2)


def hierarchical_search(frame, ref):
    if ref.shape[0] <= 8 or ref.shape[1] <= 8:
        return exhaustive_search2(frame, ref)
    new_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    new_frame = cv2.pyrDown(new_frame)
    new_ref = cv2.GaussianBlur(ref, (5, 5), 0)
    new_ref = cv2.pyrDown(new_ref)
    print(frame.shape, ref.shape, new_frame.shape, new_ref.shape)
    x, y = hierarchical_search(new_frame, new_ref)
    print(x, y)
    best = -math.inf
    for i in range(2 * x - 1, 2 * x + 1):
        for j in range(2 * y - 1, 2 * y + 1):
            if 0 <= i - ref.shape[0] / 2 < ref.shape[0] and 0 <= j - ref.shape[1] / 2 < ref.shape[1]:
                temp = np.sum(frame[int(i - ref.shape[0] / 2): int(i + (ref.shape[0] - 1) / 2),\
                                int(j - ref.shape[1] / 2): int(j + (ref.shape[1] - 1) / 2)] * ref)
                if temp > best:
                    best = temp
                    argbest = i, j
    return 2 * x + argbest[0], 2 * y + argbest[1]


def logarithmic_search(frame, ref):
    l = int(ref.shape[1]/2)
    for i in range(-1, 1):
        for j in range(-1, 1):
            pass
    
    

def search(frame, ref, x, y, p, method):
    threshold = lambda x : 0 if x < 0 else x
    xt, yt = method(frame[threshold(x - p): threshold(x + p), threshold(y - p): threshold(y + p)], ref)
    return threshold(x - p) + xt, threshold(y - p) + yt


cap = cv2.VideoCapture('movie.mov')
ref = cv2.imread('reference.jpg')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mov',fourcc, cap.get(cv2.CAP_PROP_FPS),\
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    exit(0)


ret, frame = cap.read()
while not ret: ret, frame = cap.read()

x, y = exhaustive_search(frame, ref)

num = 1
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        x, y = search(frame, ref, x, y, P, hierarchical_search)
        #x, y = exhaustive_search(frame, ref)
        num+=1
        frame = cv2.rectangle(frame,(int(y  - ref.shape[1]/2), int(x - ref.shape[0]/2)), \
                          (int(y + ref.shape[1]/2), int(x + ref.shape[0]/2)), (0, 0, 255), 3)
        _debug_print(frame)
        #out.write(frame)
        break
    else: break


cap.release()
out.release()