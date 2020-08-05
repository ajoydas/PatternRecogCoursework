import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

window_size = 7

vidcap = cv2.VideoCapture('movie.mov')

success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("D:/cv2out/frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    # print('Read a new frame: ', success)
    count += 1

reference = cv2.imread('reference.jpg', 0)
image_read = cv2.imread('D:/cv2out/frame500.jpg', 0)
w, h = reference.shape[::-1]

print(w, h)
print(reference.shape)


# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# for meth in methods:
def find_diff(theImage, reference, loop_i, loop_j):
    # cv2.imshow(theImage[280:340, 330:390,])
    max_diff = -math.inf
    max_i = 0
    max_j = 0
    for j in range(loop_j):
        for i in range(loop_i):
            im3 = theImage[i:i + h, j:j + w]
            diff = (np.sum((im3 - reference) ** 2))
            if diff > max_diff:
                max_diff = diff
                max_i = i
                max_j = j
    return max_i, max_j


def find_diff_framed(theImage, reference, loop_i, loop_j, a_i, a_j, ws):


    w, h = reference.shape[::-1]

    ww, hh = theImage.shape[::-1]

    max_diff = -math.inf

    max_i = 0
    max_j = 0
    for j in range(math.ceil(a_j - ws), math.ceil(a_j + ws + 1)):
        for i in range(math.ceil(a_i - ws), math.ceil(a_i + ws + 1)):
            im3 = theImage[i:i + h, j:j + w]
            # print(im3.shape[0], im3.shape[1], "ww")
            diff = (np.sum((im3 - reference) ** 2))
            if diff > max_diff:
                max_diff = diff
                max_i = i
                max_j = j
    return max_i, max_j


print(image_read.shape)
# im3 = image_read[0:h, 0:w]

loop_i = image_read.shape[0] - h
loop_j = image_read.shape[1] - w
print(reference.shape)

print(loop_i, loop_j)
# video = cv2.VideoWriter('video.avi',-1,1,(image_read.shape[0],image_read.shape[1]))
ws = window_size
for i in range(count):
    image_read = cv2.imread('D:/cv2out/frame%d.jpg' % i, 0)
    if i == 0:
        a_i, a_j = find_diff(image_read, reference, loop_i, loop_j)
    else:
        a_i, a_j = find_diff_framed(image_read, reference, loop_i, loop_j, a_i, a_j, ws)

    a_i = math.ceil(a_i)
    a_j = math.ceil(a_j)
    cv2.rectangle(image_read, (a_j, a_i), (a_j + w, a_i + h), (178, 34, 34), 2)
    cv2.imwrite('D:/cv2out/kaj_kore%d.jpg' % i, image_read)


