import numpy as np
import time
import cv2
import cv2.aruco as aruco
import os
from matplotlib import pyplot as plt
from numpy import zeros, float32, mgrid
from PIL import Image
import imgviz


img_name = '0032'
imgs_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/dalian_xinghaiwandaqiao_video_input_rgb/outputs/labelme_outputs/JPEGImages'
detected_mask_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/dalian_xinghaiwandaqiao_video_input_rgb/outputs/labelme_outputs/SegmentationClass'
save_dir = os.path.join(detected_mask_dir.rsplit('/', 1)[0], 'drawn_marker_and_detected_masks')
os.makedirs(save_dir, exist_ok=True)



detected_mask_pil = Image.open(os.path.join(detected_mask_dir, img_name+'.png'))
print('detected_mask_pil.mode:', detected_mask_pil.mode)
detected_mask_np = np.array(detected_mask_pil)
print('detected_mask_np.shape:' , detected_mask_np.shape)

detected_mask_pil_rgb = detected_mask_pil.convert('RGB')
detected_mask_np_rgb = np.array(detected_mask_pil_rgb)

print(detected_mask_np_rgb.shape, np.unique(detected_mask_np_rgb), detected_mask_pil_rgb.mode)

print(detected_mask_np.shape, np.unique(detected_mask_np), detected_mask_pil.mode)
############################ 先通过相机图像标定相机 ########################################
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
    # plt.figure(figsize=(12,12))
    # plt.axis("off")
    # plt.imshow(img)
    # plt.show()
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return ret,mtx,dist,rvecs,tvecs

_,mtx,dist,_,_ = calcam(cbfile1,6)
# print(ret,mtx,dist,rvecs,tvecs)
# print('ret:', ret)
print('mtx:\n', mtx)
print('dist:\n', dist, np.array(dist).shape)
# print('rvecs:\n', rvecs)
# print('tvecs:\n', tvecs)




########################################################################################
#读取图片
# frame=cv2.imread('demo.png')
# frame = cv2.imread(rf'D:\project\code\markers_tools\ArUco\armark_new_size=540\aruco_marker_0.png')
# frame = cv2.imread(rf"D:\Downloads\image_aruco.png")
# frame = cv2.imread(rf"D:\Downloads\Snipaste_2025-06-10_15-32-07.jpg")

frame = cv2.imread(os.path.join(imgs_dir, img_name+'.jpg'))
#调整图片大小
# frame=cv2.resize(frame,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_CUBIC)
frame=cv2.resize(frame,None,fx=1,fy=1,interpolation=cv2.INTER_CUBIC)
#灰度话
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#设置预定义的字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
#使用默认值初始化检测器参数
parameters =  aruco.DetectorParameters()
#使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
print("corners:",corners, corners[0].shape)
# print("ids:",ids)
# print("rejectedImgPoints:",rejectedImgPoints, len(rejectedImgPoints), rejectedImgPoints[0].shape)
#画出标志位置
#    如果找不到id

# dist=np.array(([[-0.51328742,  0.33232725 , 0.01683581 ,-0.00078608, -0.1159959]]))
# #
# mtx=np.array([[464.73554153, 0.00000000e+00 ,323.989155],
#  [  0.,         476.72971528 ,210.92028],
#  [  0.,           0.,           1.        ]])

font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)

if ids is not None:

    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

    # 估计每个标记的姿态并返回值rvet和tvec ---不同
    # from camera coeficcients
    (rvec-tvec).any() # get rid of that nasty numpy value array error

#        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #绘制轴
#        aruco.drawDetectedMarkers(frame, corners) #在标记周围画一个正方形

    for i in range(rvec.shape[0]):
        # cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
        aruco.drawDetectedMarkers(frame, corners)
    ###### DRAW ID #####
    # cv2.putText(frame, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
    print("Id: ", ids, 'rvec:', rvec, 'tvec:', tvec)
    print('\n')

else:
    ##### DRAW "NO IDS" #####
    cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)




# aruco.drawDetectedMarkers(frame, corners,ids, borderColor=(0, 255, 0))
aruco.drawDetectedMarkers(detected_mask_np_rgb, corners, borderColor=(0, 255, 0))

print('frame, frame.shape:', frame.shape, type(frame))
# cv2.imshow("frame",frame)
# cv2.waitKey(0)



def colored_mask(mask, save_path=None):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    # print(colormap, type(colormap), colormap.flatten(), colormap.shape)
    lbl_pil.putpalette(colormap.flatten())
    if save_path is not None:
        lbl_pil.save(save_path)

    return lbl_pil  


detected_mask_pil_rgb = Image.fromarray(detected_mask_np_rgb)
# detected_mask_np_rgb = np.array(detected_mask_pil_rgb)

# detected_mask_pil_p = detected_mask_pil_rgb.convert('P')
# detected_mask_np_p = np.array(detected_mask_pil_p)
# detected_mask_pil_p = colored_mask(detected_mask_np_p, save_path=os.path.join(save_dir, img_name+'.png'))
# detected_mask_pil_p.save(os.path.join(save_dir, img_name+'.png'))
# detected_mask_np_p = np.array(detected_mask_pil_p)
cv2.imshow("detected_mask_np_rgb", detected_mask_np_rgb[:, :, ::-1])
cv2.imwrite(save_dir+'/'+img_name+'.png', detected_mask_np_rgb[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
