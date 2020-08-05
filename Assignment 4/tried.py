import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

window_size = 16

vidcap = cv2.VideoCapture('movie.mov')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
global fps
fps = vidcap.get(cv2.CAP_PROP_FPS)

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
    # cv2.imshow(theImage[280:340, 330:390,])
    # windowed_loop_i_s = a_i - window_size
    # windowed_loop_i_e = a_i + window_size
    # if windowed_loop_i_s>loop_i:
    #   windowed_loop_i_s=loop_i
    # if windowed_loop_i_s<0:
    #   windowed_loop_i_s=0
    # if windowed_loop_i_e > loop_i:
    #   windowed_loop_i_e = loop_i
    # if windowed_loop_i_e < 0:
    #   windowed_loop_i_e = 0

    # windowed_loop_j_s = a_j - window_size
    # windowed_loop_j_e = a_j + window_size
    # if windowed_loop_j_s>loop_j:
    #   windowed_loop_j_s=loop_j
    # if windowed_loop_j_s<0:
    #   windowed_loop_j_s=0
    # if windowed_loop_j_e > loop_j:
    #   windowed_loop_j_e = loop_j
    # if windowed_loop_j_e < 0:
    #   windowed_loop_j_e = 0
    w, h = reference.shape[::-1]
    # print(w, h, 'what?')
    ww, hh = theImage.shape[::-1]
    # print(ww, hh, 'whatge?')
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


def find_diff_framed_log2d(theImage, reference, loop_i, loop_j, a_i, a_j, ws):
    w, h = reference.shape[::-1]
    k = math.ceil(math.log(window_size, 2))
    d = 2 ** (k - 1)
    if ws < 1:
        return a_i, a_j
    # print(d, ' ', k)
    max_diff = -math.inf
    i_arr = [math.ceil(a_i), math.ceil(a_i - d), math.ceil(a_i + d)]
    j_arr = [math.ceil(a_j), math.ceil(a_j - d), math.ceil(a_j + d)]

    max_i = 0
    max_j = 0
    for j in range(len(j_arr)):
        for i in range(len(i_arr)):
            im3 = theImage[i_arr[i]:i_arr[i] + h, j_arr[j]:j_arr[j] + w]

            diff = (np.sum((im3 - reference) ** 2))
            if diff > max_diff:
                max_diff = diff
                max_i = i_arr[i]
                max_j = j_arr[j]

    return find_diff_framed_log2d(theImage, reference, loop_i, loop_j, max_i, max_j, ws / 2)


def find_diff_framed_hierarchical(theImage, reference, loop_i, loop_j, a_i, a_j, ws):
    # theImage_s = theImage.copy()
    # reference_s = reference.copy()

    theImage_l1 = cv2.blur(theImage, (3, 3))
    reference_l1 = cv2.blur(reference, (3, 3))
    theImage_l1 = theImage_l1[::2, ::2]
    reference_l1 = reference_l1[::2, ::2]
    theImage_l2 = cv2.blur(theImage_l1, (3, 3))
    reference_l2 = cv2.blur(reference_l1, (3, 3))
    theImage_l2 = theImage_l2[::2, ::2]
    reference_l2 = reference_l2[::2, ::2]

    #
    # # print(theImage.shape, theImage_l1.shape, theImage_l2.shape)
    # #
    x1, y1 = find_diff_framed_log2d(theImage_l2, reference_l2, loop_i, loop_j, a_i // 4, a_j // 4, ws / 4)

    x1 = x1 - a_i // 4
    y1 = y1 - a_j // 4

    x2, y2 = find_diff_framed(theImage_l1, reference_l1, loop_i, loop_j, a_i // 2 + 2 * x1, a_j // 2 + 2 * y1, ws / 2)

    x2 = x2 - a_i // 2
    y2 = y2 - a_j // 2

    fin_x, fin_y = find_diff_framed_log2d(theImage, reference, loop_i, loop_j, a_i + 2 * x2, a_j + 2 * y2, 1)

    return fin_x, fin_y


print(image_read.shape)
# im3 = image_read[0:h, 0:w]

loop_i = image_read.shape[0] - h
loop_j = image_read.shape[1] - w
print(reference.shape)

print(loop_i, loop_j)
# video = cv2.VideoWriter('video.avi',-1,1,(image_read.shape[0],image_read.shape[1]))
ws = window_size

height, width = image_read.shape
output_exhaustive = cv2.VideoWriter('output_ex.mov', fourcc, fps, (width, height))
output_log2d = cv2.VideoWriter('output_l2d.mov', fourcc, fps, (width, height))
output_hr = cv2.VideoWriter('output_hr.mov', fourcc, fps, (width, height))


def template_mather(random_string):
    for i in range(count):
        image_read = cv2.imread('D:/cv2out/frame%d.jpg' % i, 0)
        if i == 0:
            a_i, a_j = find_diff(image_read, reference, loop_i, loop_j)
        else:
            if random_string == 'ex':
                a_i, a_j = find_diff_framed(image_read, reference, loop_i, loop_j, a_i, a_j, ws)
            elif random_string == 'l2d':
                a_i, a_j = find_diff_framed_log2d(image_read, reference, loop_i, loop_j, a_i, a_j, ws)
            elif random_string == 'hr':
                a_i, a_j = find_diff_framed_hierarchical(image_read, reference, loop_i, loop_j, a_i, a_j, ws)
        a_i = math.ceil(a_i)
        a_j = math.ceil(a_j)
        cv2.rectangle(image_read, (a_j, a_i), (a_j + w, a_i + h), (178, 34, 34), 2)
        cv2.imwrite('D:/cv2out/kaj_kore%d.jpg' % i, image_read)
        image_r = cv2.imread("D:/cv2out/kaj_kore%d.jpg" % i)

        if random_string == 'ex':
            output_exhaustive.write(image_r)
        elif random_string == 'l2d':
            output_log2d.write(image_r)
        elif random_string == 'hr':
            output_hr.write(image_r)

    if random_string == 'ex':
        output_exhaustive.release()
    elif random_string == 'l2d':
        output_log2d.release()
    elif random_string == 'hr':
        output_hr.release()

template_mather('ex')
template_mather('l2d')
template_mather('hr')