import cv2
import matplotlib.pyplot as plt
from numpy import zeros, float32, mgrid
import numpy as np
import os

# cbfile1 = '/home/aistudio/work/GP100.jpg'
# cbfile2 = '/home/aistudio/work/chessb1.jpg'
# cbfile1 = rf"D:\Downloads\calib.io_checker_200x150_8x11_15.png"
# cbfile1 = rf"D:\Downloads\20250603131024.jpg"

cbfile1 = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/tools/calib.io_checker_200x150_8x11_15.png'

CORNER_NUM     = (10,7) # 11*8是10，7

def calcam(cbfile, gridsize=5):
    img = cv2.imread(cbfile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CORNER_NUM, None)

    if ret:
        cv2.drawChessboardCorners(img, CORNER_NUM, corners, ret)



    obj_p = zeros((CORNER_NUM[0] * CORNER_NUM[1], 3), float32)
    obj_p[:,:2] = (mgrid[0:CORNER_NUM[0], 0:CORNER_NUM[1]].T.reshape(-1,2))*gridsize
    obj_points = []
    obj_points.append(obj_p.astype(np.float32))
    img_points = []
    img_points.append(corners.astype(np.float32))

    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(obj_points, img_points, CORNER_NUM, None, None)
    
    # plt.clf()
    plt.figure(figsize=(12,12))
    plt.axis("off")
    plt.imshow(img)
    plt.show()
    return ret,mtx,dist,rvecs,tvecs

ret,mtx,dist,rvecs,tvecs = calcam(cbfile1,6)
# print(ret,mtx,dist,rvecs,tvecs)
print('ret:', ret)
print('mtx:\n', mtx)
print('dist:\n', dist, np.array(dist).shape)
print('rvecs:\n', rvecs)
print('tvecs:\n', tvecs)