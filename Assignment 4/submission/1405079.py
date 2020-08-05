# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

def make_video(output):
    global vidcap
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    height, width, layers = output[0].shape
    outputV = cv2.VideoWriter('output.mov', fourcc, fps, (width, height))
    for frame in output:
        outputV.write(frame)
    outputV.release()


def implot(image):
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
  
def input_video():    
    global ref, frames, output, count, vidcap
    frames = []
    output = []

    vidcap = cv2.VideoCapture('movie.mov')
    success, image = vidcap.read()
    count = 0
    
    
    while success:
        output.append(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frames.append(image)
        success, image = vidcap.read()
        count += 1
    
    ref = cv2.imread("reference.jpg")
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    
    print(ref.shape)
    print(frames[0].shape)
    print("Total frames = ", count)
    
    implot(frames[0])
    implot(ref)
    
    global fx, fy
    fx, fy =frames[0].shape
    global refx, refy 
    refx, refy= ref.shape


def valid(i, j):
    return 0 <= i and i+refx < fx and 0 <= j and j+refy < fy


def sampling(image):
    percent = 0.5
    len_x, len_y = image.shape
    size = (int(len_x * percent), int(len_y * percent))
    return cv2.resize(image, size)


def hierarchical():
    global count,limit, frames, ref
    count = 0
    
    x, y = full_exhaust(frames[0], ref)
    for f in range(1,limit):
        img = frames[f].copy()
        refer = ref.copy()
        filter_size = (3,3)
           
        level1frame = sampling(cv2.GaussianBlur(img, filter_size, 0))
        level1ref = sampling(cv2.GaussianBlur(refer, filter_size, 0))
        
        level2frame = sampling(cv2.GaussianBlur(level1frame, filter_size, 0))
        level2ref = sampling(cv2.GaussianBlur(level1ref, filter_size, 0))
        
        max_x, max_y = part_exhaust(level2frame, level2ref, x//4, y//4, p//4)    
        
    #    max_x = max_x - x // 4
    #    max_y = max_y - y // 4
    #    
        max_x, max_y = part_exhaust(level1frame, level1ref, x//2 + 2* max_x, y//2 + 2*max_y, 1)
        
    #    max_x = max_x - x // 2
    #    max_y = max_y - y // 2
    #    
        x, y = part_exhaust(frames[f], ref, x + 2* max_x, y + 2*max_y, 1)
        cv2.rectangle(output[f],(int(y), int(x)), \
                              (int(y + ref.shape[1]), int(x + ref.shape[0])), (0, 0, 255), 3)
    
    print("hierarchical: "+str(count/len(frames)))
    make_video(output)



def log2dSearch():
    global count, limit
    count = 0
    x, y = full_exhaust(frames[0], ref)
    for f in range(1,limit):
        sp = p
        
        max_val = -np.inf
        max_x = x
        max_y = y
        
        k = np.ceil(np.log2(sp))
        d = int(2**(k-1))
        
        while d > 1:
            for j in range(max_y - sp, max_y + sp +1, d):
                for i in range(max_x - sp, max_x + sp + 1, d):
                    if valid(i, j) == False:
                        continue
                    
                    portion = frames[f][i:i + refx, j:j + refy]
                    diff = (np.sum((portion - ref) ** 2))
                    if diff > max_val:
                        max_val = diff
                        max_x = i
                        max_y = j
                    count +=1
    
            k = np.ceil(np.log2(sp))
            d = int(2**(k-1))
            sp = sp//2
    
        x = max_x
        y = max_y
        
        cv2.rectangle(output[f],(int(y), int(x)), \
                              (int(y + ref.shape[1]), int(x + ref.shape[0])), (0, 0, 255), 3)
    
    print("log2dSearch: "+str(count/len(frames)))
    make_video(output)


def full_exhaust(given, ref):
    global count
    max_val = -np.inf
    max_x = 0
    max_y = 0
    for j in range(fx-refx):
        for i in range(fy-refy):
#            print(i,j)
            if valid(i, j) == False:
                continue
            portion = given[i:i + refx, j:j + refy]
            diff = (np.sum((portion - ref) ** 2))
            if diff > max_val:
                max_val = diff
                max_x = i
                max_y = j
            count +=1
            
    return max_x, max_y

def part_exhaust(given, ref, prev_x, prev_y, p):
    global count
    f_x, f_y = given.shape
    ref_x, ref_y = ref.shape
    max_val = -np.inf
    max_x = 0
    max_y = 0
    for j in range(prev_y-p, prev_y + p + 1):
        for i in range(prev_x - p, prev_x + p + 1):
            if (0 <= i and i+ref_x < f_x and 0 <= j and j+ref_y < f_y) == False:
                continue
            
            portion = given[i:i + ref_x, j:j + ref_y]
            diff = (np.sum((portion - ref) ** 2))
            if diff > max_val:
                max_val = diff
                max_x = i
                max_y = j
            count +=1
            
    return max_x, max_y

def exhaustive():
    global count, limit
    count = 0;
    x, y = full_exhaust(frames[0], ref)
    for f in range(1,limit):
        x, y = part_exhaust(frames[f], ref, x, y, p)
        cv2.rectangle(output[f],(int(y), int(x)), \
                              (int(y + ref.shape[1]), int(x + ref.shape[0])), (0, 0, 255), 3)

    print("exhaustive: "+str(count/len(frames)))        
    make_video(output)



input_video()
global p
p = 50
limit = 200
exhaustive()

input_video()
log2dSearch()

input_video()
hierarchical()



    
    

























