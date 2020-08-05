# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
vidcap = cv2.VideoCapture('movie.mov')
success, image = vidcap.read()
count = 0
frames = []
actual = []
actual2 = []
actual3 = []
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
global fps
fps = vidcap.get(cv2.CAP_PROP_FPS)

while success:
    actual.append(image)
    actual2.append(image)
    actual3.append(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)
    image = (image * 0.0005)
    frames.append(image)
    success, image = vidcap.read()
    count += 1
ref = cv2.imread("reference.jpg", 0)
ref = cv2.bitwise_not(ref)
ref = ref*0.0005
print(ref.shape)
print(frames[0].shape)
print("no of frames = ", count)
global P
P = 200


def valid(img, i, j):
    return 0 <= i < len(img) and 0 <= j < len(img[0])

def exhaust(frames, ref, new_frames):
    ct = 0
    op_count = 0
    for fr in range(len(frames)):
        img = cv2.filter2D(frames[fr], -1, ref)
        implot(img)
        if ct % 100 == 0:
            print(ct)
        val = 0
        global l, r, u, d, pos_x, pos_y
        l = 0
        r = len(img[0])
        u = 0
        d = len(img)
        print(str(l)+" "+ str(r)+" "+ str(u)+" "+ str(d))
        return
        if fr > 0:
            l = pos_y - P
            u = pos_x - P
            r = pos_y + P
            d = pos_x + P
        for i in range(u, d + 1):
            for j in range(l, r + 1):
                if not valid(img, i, j):
                    continue
                op_count += 1
                if img[i][j] >= val:
                    val = img[i][j]
                    pos_x = i
                    pos_y = j
        cv2.rectangle(new_frames[ct], (pos_y - len(ref[0]) // 2, pos_x - len(ref) // 2),
                      (pos_y + len(ref[0]) - len(ref[0]) // 2, pos_x + len(ref) - len(ref) // 2), color=(255, 255, 0),
                      thickness=2)
        ct += 1
    return new_frames, op_count/len(frames)

frames1, cnt1 = exhaust(frames, ref,actual)

height, width, layers = frames1[0].shape
output = cv2.VideoWriter('output.mov', fourcc, fps, (width, height))
for frame in frames1:
    output.write(frame)
output.release()


























