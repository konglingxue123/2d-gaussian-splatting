import json
from PIL import Image
import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw
import cv2
from matplotlib import pyplot as plt
import os
import cv2.aruco as aruco
import os
from matplotlib import pyplot as plt
from numpy import zeros, float32, mgrid
import imgviz

def colored_mask(mask, save_path=None):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    # print(colormap, type(colormap), colormap.flatten(), colormap.shape)
    lbl_pil.putpalette(colormap.flatten())
    if save_path is not None:
        lbl_pil.save(save_path)

    return lbl_pil  

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


# img_name = '0001'
img_name = '0032'
colmap_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/dalian_xinghaiwandaqiao_video_input_rgb'
render_mesh_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/dalian_xinghaiwandaqiao_rgb'
save_output_mesh_ray_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/dalian_xinghaiwandaqiao_rgb/based_on_2d_npy_file_edge'
os.makedirs(save_output_mesh_ray_dir, exist_ok=True)
org_img_dir = f'/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/dalian_xinghaiwandaqiao_video_input_rgb/outputs/labelme_outputs/JPEGImages'



frame = cv2.imread(os.path.join(org_img_dir, img_name+'.jpg'))
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


##########################################################################################################################################################################
num_classes = 3

if num_classes == 1:
    class_colors_list = [[128, 0, 0]]
elif num_classes == 2:
    class_colors_list = [[128, 0, 0], [0, 128, 0]]
elif num_classes == 3:
    class_colors_list = [[128, 0, 0], [0, 128, 0], [0, 0, 128]]
else:
    raise ValueError('num_classes should be 1, 2 or 3, 1=crack, 2=spalling, 3=corrosion')

#########################  贴图裂缝骨架   ###################################

dilate_size = 1
# add_xy1 = [900, 600]
# add_xy2 = [650, 650]

img_dir = f'{colmap_dir}/images/{img_name}.jpg'

im = Image.open(img_dir)
# 在im上绘制绿色的4个角点
draw = ImageDraw.Draw(im)
draw.point([(corners[0][0][0][0], corners[0][0][0][1]), 
            (corners[0][0][1][0], corners[0][0][1][1]), 
            (corners[0][0][2][0], corners[0][0][2][1]),
            (corners[0][0][3][0], corners[0][0][3][1])], fill='green')
# draw.rectangle((np.min(corners[0][0][:, 0], axis=0), np.min(corners[0][0][:, 1], axis=0), np.max(corners[0][0][:, 0], axis=0), np.max(corners[0][0][:, 1], axis=0)), outline='red')
# draw.rectangle((add_xy1[0]-dilate_size, add_xy1[1]-dilate_size, add_xy1[0]+dilate_size, add_xy1[1]+dilate_size), outline='green')
im_np = np.array(im)

def merge_imgs_and_masks(img_dir, mask_dir, color_list = [[128, 0, 0]]):
    # os.makedirs(save_dir, exist_ok=True)
    
    n = 0
    
    for img_name_with_ext in os.listdir(img_dir):
        n += 1
        # img_name = img_name_with_ext.split('.')[0]

        img = os.path.join(img_dir, img_name + '.jpg')
        mask = os.path.join(mask_dir, img_name + '.png')

        img_pil = Image.open(img)
        mask_pil = Image.open(mask)

        img_arr = np.array(img_pil)
        mask_arr = np.array(mask_pil)
        print('img_arr.shape, mask_arr.shape:', img_arr.shape, mask_arr.shape)

        # 将mask中值为1的像素点的位置在img的位置上进行融合
        for id, color in enumerate(color_list):
            img_arr[..., 0][mask_arr == id+1] = color[0]
            img_arr[..., 1][mask_arr == id+1] = color[1]
            img_arr[..., 2][mask_arr == id+1] = color[2]

        # 保存融合后的图片
        img_merged = Image.fromarray(img_arr)
        # img_merged.save(os.path.join(save_dir, img_name + f'{ext}'))
    return img_arr, img_merged

def merge_imgs_pil_and_masks_dir(img_pil, img_name, mask_dir, color_list = [[128, 0, 0]]):
    # os.makedirs(save_dir, exist_ok=True)
    
    # n = 0
    
    # for img_name_with_ext in os.listdir(img_dir):
    #     n += 1
        # img_name = img_name_with_ext.split('.')[0]

    # img = os.path.join(img_dir, img_name + '.jpg')
    mask = os.path.join(mask_dir, img_name + '.png')

    # img_pil = Image.open(img)
    mask_pil = Image.open(mask)

    img_arr = np.array(img_pil)
    mask_arr = np.array(mask_pil)
    print('img_arr.shape, mask_arr.shape:', img_arr.shape, mask_arr.shape)

    # 将mask中值为1的像素点的位置在img的位置上进行融合
    for id, color in enumerate(color_list):
        img_arr[..., 0][mask_arr == id+1] = color[0]
        img_arr[..., 1][mask_arr == id+1] = color[1]
        img_arr[..., 2][mask_arr == id+1] = color[2]

    # 保存融合后的图片
    img_merged = Image.fromarray(img_arr)
    # img_merged.save(os.path.join(save_dir, img_name + f'{ext}'))
    return img_arr, img_merged


def merge_imgs_pil_and_masks_np(img_pil, img_name, mask_np, color_list = [[128, 0, 0]]):
    # os.makedirs(save_dir, exist_ok=True)
    
    # n = 0
    
    # for img_name_with_ext in os.listdir(img_dir):
    #     n += 1
        # img_name = img_name_with_ext.split('.')[0]

    # img = os.path.join(img_dir, img_name + '.jpg')
    # mask = os.path.join(mask_dir, img_name + '.png')

    # img_pil = Image.open(img)
    # mask_pil = Image.open(mask)

    img_arr = np.array(img_pil)
    mask_arr = mask_np
    print('img_arr.shape, mask_arr.shape:', img_arr.shape, mask_arr.shape)

    # 将mask中值为1的像素点的位置在img的位置上进行融合
    for id, color in enumerate(color_list):
        img_arr[..., 0][mask_arr == id+1] = color[0]
        img_arr[..., 1][mask_arr == id+1] = color[1]
        img_arr[..., 2][mask_arr == id+1] = color[2]

    # 保存融合后的图片
    img_merged = Image.fromarray(img_arr)
    # img_merged.save(os.path.join(save_dir, img_name + f'{ext}'))
    return img_arr, img_merged

# 贴图裂缝
# crack_mask_magma_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/projection/CFD_044_dist_on_skel_lee_pil_for_projection_magma.png'
mask_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/dalian_xinghaiwandaqiao_video_input_rgb/outputs/labelme_outputs/SegmentationClass'
# mask_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/dalian_xinghaiwandaqiao_video_input_rgb/outputs/labelme_outputs/drawn_marker_and_detected_masks'
crack_mask_magma_dir = f'{mask_dir}/{img_name}.png'
# crack_mask_magma_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/projection/CFD_044.jpg'
crack_mask_magma_pil_l = Image.open(crack_mask_magma_dir)
print('转换l前的np.unique(crack_mask_magma_pil_l_np)', np.unique(np.array(crack_mask_magma_pil_l)))
marker_point_zero_np = np.zeros((crack_mask_magma_pil_l.size[1], crack_mask_magma_pil_l.size[0])).astype(np.uint8)
print('marker_point_zero_np.shape', marker_point_zero_np.shape, np.unique(marker_point_zero_np))

print('corners[0][0].shape', corners[0][0].shape)
for i in range(corners[0][0].shape[0]):
    # i = int(i)
    # print(i, corners[0][0][i][0], corners[0][0][i][1])
    marker_point_zero_np[int(corners[0][0][i][1]), int(corners[0][0][i][0])] = 1
# marker_point_zero_np[corners[0][0][0][1], corners[0][0][0][0]] = 1
# marker_point_zero_np[corners[0][0][1][1], corners[0][0][1][0]] = 1
# marker_point_zero_np[corners[0][0][2][1], corners[0][0][2][0]] = 1
# marker_point_zero_np[corners[0][0][3][1], corners[0][0][3][0]] = 1
###################################

# 在im上绘制绿色的4个角点(用于在3D模型上显示检测到的4个角点)
# draw_mask_marker = ImageDraw.Draw(crack_mask_magma_pil_l)
# draw_mask_marker.point([(corners[0][0][0][0], corners[0][0][0][1]), 
#             (corners[0][0][1][0], corners[0][0][1][1]), 
#             (corners[0][0][2][0], corners[0][0][2][1]),
#             (corners[0][0][3][0], corners[0][0][3][1])], fill='green')
###################################

# crack_mask_magma_pil_l = crack_mask_magma_pil_l.convert('L')

crack_mask_magma_pil_l_np = np.zeros((crack_mask_magma_pil_l.size[1], crack_mask_magma_pil_l.size[0])).astype(np.uint8)

############### 读取npy文件，并将其转换为图片 #############
cls_num_edge_pair_dict_npy_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/dalian_xinghaiwandaqiao_rgb/img_point_width/edge_skel_edge_pair/cls_num_edge_pair_dict_0032.npy'
cls_num_edge_to_calculate_mid_axis_dict_npy_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/dalian_xinghaiwandaqiao_rgb/img_point_width/edge_skel_edge_pair/cls_num_edge_to_calculate_mid_axis_dict_0032.npy' 
cls_num_skel_calculate_mid_axis_dict_npy_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/dalian_xinghaiwandaqiao_rgb/img_point_width/edge_skel_edge_pair/cls_num_skel_calculate_mid_axis_dict_0032.npy' 

cls_num_edge_pair_dict = np.load(cls_num_edge_pair_dict_npy_dir, allow_pickle=True).item()
cls_num_edge_to_calculate_mid_axis_dict = np.load(cls_num_edge_to_calculate_mid_axis_dict_npy_dir, allow_pickle=True).item()
cls_num_skel_calculate_mid_axis_dict = np.load(cls_num_skel_calculate_mid_axis_dict_npy_dir, allow_pickle=True).item()

# 类别
num_classes = 3
################ 通过图像计算保存的npy文件中，所有的边界，但是和下面的中线和边界的交点不一样 #####################
def get_crack_mask_np_from_any_edge(num_classes, crack_mask_magma_pil_l_np, cls_num_edge_to_calculate_mid_axis_dict):
    for cls in range(1, num_classes+1):
        try:
            crack_edge_list = cls_num_edge_to_calculate_mid_axis_dict[cls]


            for crack_edge in crack_edge_list:
                # print(type(crack_edge))
                crack_edge = crack_edge.astype(int)
                # crack_mask_magma_pil_l_np中，在crack_edge一系列点的位置坐标处，设置为255
                crack_mask_magma_pil_l_np[crack_edge[:, 0], crack_edge[:, 1]] = cls
        
        except:
            print(f"No crack edge found for class={cls} in img_name={img_name}")
            continue
    return crack_mask_magma_pil_l_np

# crack_mask_magma_pil_l_np = get_crack_mask_np_from_any_edge(num_classes, crack_mask_magma_pil_l_np, cls_num_edge_to_calculate_mid_axis_dict)

################ 通过图像计算保存的npy文件中，对中线和边界的交点（更少点， 按照边界对2个点的顺序，便于计算3d裂缝宽度） #####################
def get_crack_mask_np_from_pair_edge(num_classes, crack_mask_magma_pil_l_np, cls_num_edge_pair_dict):
    x_order_list = []
    y_order_list = []
    for cls in range(1, num_classes+1):
        try:
            crack_edge_list = cls_num_edge_pair_dict[cls]
            # print(crack_edge_list)

            for crack_edge in crack_edge_list:
                # print(type(crack_edge))
                crack_edge = crack_edge.astype(int)
                # crack_mask_magma_pil_l_np中，在crack_edge一系列点的位置坐标处，设置为255
                crack_mask_magma_pil_l_np[crack_edge[:, 0], crack_edge[:, 1]] = cls
                crack_mask_magma_pil_l_np[crack_edge[:, 2], crack_edge[:, 3]] = cls
                x_order_list.append(crack_edge[:, 0])
                x_order_list.append(crack_edge[:, 2])
                y_order_list.append(crack_edge[:, 1])
                y_order_list.append(crack_edge[:, 3])
                # y_order_list.append(crack_edge[:, 1], crack_edge[:, 3])
        
        except:
            print(f"No crack edge found for class={cls} in img_name={img_name}")
            continue
    return crack_mask_magma_pil_l_np, x_order_list, y_order_list

crack_mask_magma_pil_l_np, x_order_list, y_order_list = get_crack_mask_np_from_pair_edge(num_classes, crack_mask_magma_pil_l_np, cls_num_edge_pair_dict)

crack_mask_pil = colored_mask(crack_mask_magma_pil_l_np)
# plt.imshow(crack_mask_pil)
# plt.show()
# crack_mask_magma_pil_l_np = np.array(crack_mask_magma_pil_l)
crack_mask_magma_pil_l_np = np.array(crack_mask_pil)
print(crack_mask_magma_pil_l_np.shape, np.unique(crack_mask_magma_pil_l_np))

# crack_mask_magma_pil = crack_mask_magma_pil_l.convert('RGB')
crack_mask_magma_pil = crack_mask_pil.convert('RGB')
crack_mask_magma_np = np.array(crack_mask_magma_pil)

print(crack_mask_magma_pil_l.mode)
print(crack_mask_magma_pil_l.size, im.size)
# if im.size != crack_mask_magma_pil_l.size:
#     crack_mask_magma_pil_l = crack_mask_magma_pil_l.resize((448, 448))  # 这里后续用检测的mask，shape与原图一样，就不用resize了
# crack_mask_magma_pil_l_np_nonzero_id = np.where(np.array(crack_mask_magma_pil_l) !=0)

# crack_mask_magma_pil_l_np = np.array(crack_mask_magma_pil_l)
crack_mask_magma_pil_l_np = np.array(crack_mask_pil)
print(crack_mask_magma_pil_l_np.shape, np.unique(crack_mask_magma_pil_l_np))

crack_mask_magma_pil = crack_mask_magma_pil_l.convert('RGB')
crack_mask_magma_np = np.array(crack_mask_magma_pil)


mask_zero_same_size, mask_zero_same_size_pil_l = crack_mask_magma_pil_l_np, crack_mask_magma_pil_l

print('mask_zero_same_size.shape', mask_zero_same_size.shape) # h, w = (1080, 1920)

plt.imshow(crack_mask_magma_pil_l_np)
plt.show()

# img_with_mask_dir = f'/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/beijing_baigeqiao_video_input_to_rgb_with_label/images/{selected_image_name}.jpg'
# result_image_magma, mask_zero_same_size_rgb_pil = merge_imgs_pil_and_masks_dir(im, img_name, mask_dir, color_list=class_colors_list)
result_image_magma, mask_zero_same_size_rgb_pil = merge_imgs_pil_and_masks_np(im, img_name, crack_mask_magma_pil_l_np, color_list=class_colors_list)
result_image_magma_pil = Image.fromarray(result_image_magma)

result_image_magma_pil = Image.fromarray(result_image_magma)

h, w = mask_zero_same_size.shape
print('h, w', h, w)

plt.subplot(1, 3, 1)
plt.imshow(result_image_magma_pil)

plt.subplot(1, 3, 2)
# plt.imshow(crack_mask_magma_pil_l)
plt.imshow(crack_mask_pil)

plt.subplot(1, 3, 3)
plt.imshow(im)

plt.show()




######################### 1. 读取json文件，并将frames_dict中包含的file_path的图像复制到新文件夹中 ##########




# draw = ImageDraw.Draw(im)
# draw.rectangle((add_xy1[0], add_xy1[1], add_xy2[0], add_xy2[1]), fill='red', outline='red')



json_dir = f'{colmap_dir}/transforms.json'
# json_dir = 'datasets/tea_pot_transform_mode/transforms_train.json'
# 读取json文件
with open(json_dir, 'r') as f:
    data = json.load(f)

# 打印数据
frames_dict = data['frames']

# 将frames_dict中包含的file_path的图像复制到新文件夹中
import os
import shutil

# new_dir = '/home/ubunto/Project/konglx/pcd/Open3D/datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/images_colmap_sparsed'
# if not os.path.exists(new_dir):
#     os.makedirs(new_dir)
    
# for frame_id in frames_dict:
    
#     file_path = frame_id['file_path'] # 这是路径
#     if os.path.exists(file_path):
#         shutil.copy(file_path, new_dir)


frames_data = data['frames']

################################ 2. 读取图像，并将红色区域的点云提取出来 ####
# load a scene point cloud
# scene = o3d.io.read_point_cloud('/home/ubunto/Project/konglx/pcd/3dgs/gaussian-splatting-main/datasets/tea_pot_colmap_this_rgba_for_sam2unet_all_pipeline_from_colmap_to_nerf_to_2dgs/colmap_sparse/0/sparse.ply')
scene = o3d.io.read_point_cloud(f'{render_mesh_dir}/train/ours_30000/fuse_post.ply')
scene_points = np.asarray(scene.points)
scene_colors = np.asarray(scene.colors)
print('scene_points.shape', scene_points.shape)
print('scene_colors.shape', scene_colors.shape)

import numpy as np
import open3d

# 图像的rgb点
# img_np = np.asarray(im) / 255.0
img_np = result_image_magma / 255.0
# img_np.shape, img_np.shape[0]*img_np.shape[1]
img_np_reshape = img_np.reshape(-1, 3)



## 原始点云 ##
pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(scene_points)

# pcd.colors = open3d.utility.Vector3dVector(colors)
intrinsics = np.array([
    [data['fl_x'], 0,            data['cx'], 0],
    [0,            data['fl_y'], data['cy'], 0],
    [0,            0,            1,          0],
    [0,            0,            0,          1]
])

###########################################
# 增加红色的新点云
# 生成网格点（注意 y 在前，x 在后，与图像的行列索引一致）---->2D网格点
y_range = slice(0, im.size[1])  # y 对应行索引
x_range = slice(0, im.size[0])  # x 对应列索引
# print(x_range)
# 生成网格点矩阵
y, x = np.mgrid[y_range, x_range]
print(x.shape)
print(y.shape)
# 组合为二维坐标点，并调整形状为 [30000, 2]
selected_area_np = np.column_stack((x.ravel(), y.ravel()))
selected_area_np_qici = np.hstack((selected_area_np, np.ones((selected_area_np.shape[0], 1))))
print(selected_area_np_qici.shape)
# 内参重复10000次，shape为[10000, 4, 4]
intrinsics_area = np.tile(intrinsics, (selected_area_np_qici.shape[0], 1, 1))
print(intrinsics_area.shape)

# 找外参数矩阵， 
# selected_image_name = 'image_54_1080'
# selected_image_name = 'image_12_240'
# selected_image_name = 'image_20_400'

# 展示特定的图像和相机位姿
for id, frame in enumerate(frames_data):
    if img_name in frame['file_path']:
        R = np.array(frame['R'])
        t = np.array(frame['t'])
        # 转置 t 并调整为列向量（3x1），然后与 r 水平拼接
        top = np.hstack([R, t.reshape(3, 1)])  # 3x4

        # 创建最后一行 [0,0,0,1]
        bottom = np.array([[0, 0, 0, 1]])

        # 垂直拼接生成 4x4 矩阵
        extrinsics = np.vstack([top, bottom])
        chosen_defect_img_id = id
        print('chosen_defect_img_id:', chosen_defect_img_id)
    else:
        continue

extrinsics_area = np.tile(extrinsics, (selected_area_np_qici.shape[0], 1, 1))
print(extrinsics_area.shape)


def get_world_coords_for_cam_centers(json_dir):# 读取json文件
    with open(json_dir, 'r') as f:
        data = json.load(f)

    extrinsics_list = []
    frames_data = data['frames']
    for frame in frames_data:
        R = np.array(frame['R'])
        t = np.array(frame['t'])
        # 转置 t 并调整为列向量（3x1），然后与 r 水平拼接
        top = np.hstack([R, t.reshape(3, 1)])  # 3x4

        # 创建最后一行 [0,0,0,1]
        bottom = np.array([[0, 0, 0, 1]])

        # 垂直拼接生成 4x4 矩阵
        extrinsics = np.vstack([top, bottom])
        extrinsics_list.append(extrinsics)
    return np.asarray(extrinsics_list)


def pixel_to_world(uv, depth, K, R, T):
    # 畸变校正（假设已校正，否则需调用cv2.undistortPoints）
    # 归一化坐标计算
    inv_K = np.linalg.inv(K)
    inv_R = np.linalg.inv(R)
    homogeneous_pixel = uv[..., np.newaxis]
    ndc = inv_K @ homogeneous_pixel  # 归一化坐标（未乘深度）
    # ndc = ndc.squeeze(-1)
    # 应用深度
    camera_coord = ndc * depth
    # camera_coord = ndc
    
    # 转换为世界坐标
    world_coord = (R.transpose(0,2,1) @ camera_coord).squeeze(-1) - \
                    (R.transpose(0,2,1) @ T[..., np.newaxis]).squeeze(-1) 
    
    # 世界坐标系下深度为1的点
    world_coords_for_depth_1 = (R.transpose(0,2,1) @ ndc).squeeze(-1) - \
                    (R.transpose(0,2,1) @ T[..., np.newaxis]).squeeze(-1) 
                    
    # # 世界坐标系下深度为0的点, 即，中心点
    # print('R.shape, T.shape:', R.shape, T.shape)
    # world_coords_for_depth_0 = -(R.transpose(0,2,1) @ T[..., np.newaxis]).squeeze(-1) 
    return world_coord, camera_coord, world_coords_for_depth_1
    # return world_coord, camera_coord, world_coords_for_depth_1, world_coords_for_depth_0
print('############ selected_area_np_qici ############', selected_area_np_qici, selected_area_np_qici.shape)
world_coords, camera_coords, world_coords_for_depth_1 = pixel_to_world(selected_area_np_qici, 0.5,  # depth=0.5是在虚拟位姿的矩形框展示
                              intrinsics_area[:,:3, :3], 
                              extrinsics_area[:, :3, :3], 
                              extrinsics_area[:,:3, 3])
# world_cam_center_coords, _, _ = pixel_to_world(selected_area_np_qici, 0.,  # depth=0.5是在虚拟位姿的矩形框展示
#                               intrinsics_area[:,:3, :3], 
#                               extrinsics_area[:, :3, :3], 
#                               extrinsics_area[:,:3, 3])
print('############ camera_coords ############', camera_coords, camera_coords.shape)
print('############ world_coords ############', world_coords, world_coords.shape)
print('############ world_coords_for_depth_1 ############', world_coords_for_depth_1, world_coords_for_depth_1.shape)
# print('############ world_cam_center_coords ############', world_cam_center_coords, world_cam_center_coords.shape)
# print('############ world_coords_for_depth_0 ############', world_coords_for_depth_0, world_coords_for_depth_0.shape)

###########################################

extrinsics_camera = get_world_coords_for_cam_centers(json_dir=json_dir)
R_cameras = extrinsics_camera[:, :3, :3]
T_cameras = extrinsics_camera[:, :3, 3]
camera_center_coords = -(R_cameras.transpose(0,2,1) @ T_cameras[..., np.newaxis]).squeeze(-1)  # 世界坐标系下深度为0的点, 即，中心点
print('############### camera_center_coords ############', extrinsics_camera.shape, R_cameras.shape, T_cameras.shape, camera_center_coords.shape)

#连接点云与相机中心点
# camera_center_to_defects_points = np.concatenate([camera_center_coords, world_coords_for_depth_1], axis=0)
############################### 3. 绘制点云与红色区域 #############

import open3d as o3d
import numpy as np
import json

WIDTH = w
HEIGHT = h


# json_dir = 'datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/transforms.json'
# # 读取json文件
# with open(json_dir, 'r') as f:
#     data = json.load(f)
# frames_data = data['frames']
# print(data['h'])

# load a scene point cloud
# scene = o3d.io.read_point_cloud('/home/ubunto/Project/konglx/pcd/projection/datasets/tea_pot_transform_mode_this_SIMPLE_PINHOLE/colmap_sparse/0/sparse.ply')
# scene = o3d.io.read_triangle_mesh('/home/ubunto/Project/konglx/pcd/image_to_3d/TRELLIS/trellis-outputs/tea-pot_letter/sample.glb')
# 可视化坐标轴. The x, y, z axis will be rendered as red, green, and blue arrows respectively.
coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])  
# coor.scale(10.0) 
vizualizer = o3d.visualization.Visualizer()
vizualizer.create_window(width=WIDTH, height=HEIGHT)

# camera_center_point_list = []
# 展示所有
for frame in frames_data:
    R = np.array(frame['R'])
    t = np.array(frame['t'])
    

    
    # 转置 t 并调整为列向量（3x1），然后与 r 水平拼接
    top = np.hstack([R, t.reshape(3, 1)])  # 3x4

    # 创建最后一行 [0,0,0,1]
    bottom = np.array([[0, 0, 0, 1]])

    # 垂直拼接生成 4x4 矩阵
    extrinsics = np.vstack([top, bottom])

    # core code. Set up a set of lines to represent the camera.
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=int(data['w']), view_height_px=int(data['h']), 
                                                                   intrinsic=intrinsics[:3,:3], extrinsic=extrinsics,
                                                                   scale=0.5)
    
    vizualizer.add_geometry(cameraLines)
    # vizualizer.add_geometry(scene)
    # vizualizer.add_geometry(coor)

# camera_center_point_list_np = np.vstack(camera_center_point_list)
# print('########## camera_center_point_list_np.shape: #############', camera_center_point_list_np.shape)


#############################################################################################################

################################### 4. 损伤mask与相机中心点的连线 #################################
# 生成网格点矩阵
marker_point_id = np.where(marker_point_zero_np !=0)
print(marker_point_id[0].shape, marker_point_id[1].shape)
x_marker_points = marker_point_id[1]
y_marker_points = marker_point_id[0]


print('np.unique(mask_zero_same_size):', np.unique(mask_zero_same_size), mask_zero_same_size.shape)


# 按照mask的标签数值进行统计
unique_mask_zero_same_size = np.unique(mask_zero_same_size)

defect_cls_dict = {}

for i in unique_mask_zero_same_size.tolist():
    if i == 0:
        continue
    else:
        defect_cls_dict[i] = []
        defect_cls_dict[i].append(np.where(mask_zero_same_size == i))
        
print('defect_cls_dict:', defect_cls_dict)
print(list(defect_cls_dict.keys()))
# print(defect_cls_dict[list(defect_cls_dict.keys())[0]])
# np.save(os.path.join(render_mesh_dir, 'defect_cls_dict.npy'), defect_cls_dict)


cls_num_of_points_and_colors_dict = {}
for i in list(defect_cls_dict.keys()):
    cls_num_of_points_and_colors_dict[i] = defect_cls_dict[i][0][0].shape[0]
    # cls_num_of_points_and_colors_dict[i].append()
print('cls_num_of_points_and_colors_dict:', cls_num_of_points_and_colors_dict)


for i in range(len(defect_cls_dict.keys())):
    x_defects_in_img = np.hstack((defect_cls_dict[list(defect_cls_dict.keys())[0]][0][1], defect_cls_dict[list(defect_cls_dict.keys())[1]][0][1]))
    y_defects_in_img = np.hstack((defect_cls_dict[list(defect_cls_dict.keys())[0]][0][0], defect_cls_dict[list(defect_cls_dict.keys())[1]][0][0]))
# print(x_order_list, y_order_list)
# x_defects_in_img = np.hstack(x_order_list)
# y_defects_in_img = np.hstack(y_order_list)
print('x_defects_in_img.shape, y_defects_in_img.shape:', x_defects_in_img.shape, y_defects_in_img.shape)

crack_mask_magma_pil_l_np_nonzero_id = np.where(mask_zero_same_size !=0)
print(crack_mask_magma_pil_l_np_nonzero_id[0].shape, crack_mask_magma_pil_l_np_nonzero_id[1].shape)
# 组合为二维坐标点，并调整形状为 [30000, 2]
# x_defects_in_img = crack_mask_magma_pil_l_np_nonzero_id[1]
# y_defects_in_img = crack_mask_magma_pil_l_np_nonzero_id[0]
# 在y_defects_in_img的基础上，增加4个角点
x_defects_in_img = np.hstack((x_defects_in_img, x_marker_points))
y_defects_in_img = np.hstack((y_defects_in_img, y_marker_points))
y_defects = y_defects_in_img   # y 对应行索引
x_defects = x_defects_in_img  # x 对应列索引
# y_defects, x_defects = np.mgrid[y_range_defects, x_range_defects]
print(x_defects.shape, x_defects)
print(y_defects.shape, y_defects)

defects_area_np = np.column_stack((x_defects.ravel(), y_defects.ravel()))
print('############ defects_area_np ############', defects_area_np, defects_area_np.shape)
defects_area_np_qici = np.hstack((defects_area_np, np.ones((defects_area_np.shape[0], 1))))
print('############ defects_area_np_qici ############', defects_area_np_qici, defects_area_np_qici.shape)


mask_zero_same_size_rgb_np = np.array(mask_zero_same_size_rgb_pil)#[:,:,::-1]  # 转换为RGB格式
print('maskzero_rgb_np.shape:', mask_zero_same_size_rgb_np, np.unique(mask_zero_same_size_rgb_np))
defects_area_mask_world_list = []
crack_mask_magma_np_color_not_zero_list = []
# print('im.size, im.size[0],im.size[1]:', im.size, im.size[0],im.size[1])
# 提取损伤和marker的3d点和颜色

for id, defect_point in enumerate(defects_area_np):
    defects_area_mask_world_list.append(world_coords[defect_point[0] + (defect_point[1]-1)*w])
    crack_mask_magma_np_color_not_zero_list.append((mask_zero_same_size_rgb_np[defect_point[1], defect_point[0], :] / 255.0).tolist())


defects_area_mask_world = np.array(defects_area_mask_world_list)
print('############ defects_area_mask_world ############', defects_area_mask_world, defects_area_mask_world.shape)

# intrinsics_defects_area = np.tile(intrinsics, (defects_area_np_qici.shape[0], 1, 1))
# extrinsics_defects_area = np.tile(extrinsics, (defects_area_np_qici.shape[0], 1, 1))
# defects_area_mask_world, camera_coords_defects, world_coords_for_depth_1_defects = pixel_to_world(defects_area_np_qici,
#                                          depth=0.5,  # depth=0.5是在虚拟位姿的矩形框展示
#                                          K=intrinsics_defects_area[:,:3, :3], 
#                                          R=extrinsics_defects_area[:, :3, :3], 
#                                          T=extrinsics_defects_area[:,:3, 3])

# 直接用wolrd_coords进行对应位置的选取

print('############ defects_area_mask_world ############', defects_area_mask_world, defects_area_mask_world.shape)

chosen_camera_center_coords = camera_center_coords[chosen_defect_img_id].reshape(-1, 3)
# print('############ chosen_camera_center_coords ############', chosen_camera_center_coords, chosen_camera_center_coords.shape)
# chosen_camera_center_coords = np.tile(chosen_camera_center_coords, (defects_area_mask_world.shape[0], 1))
print('############ chosen_camera_center_coords ############', chosen_camera_center_coords, chosen_camera_center_coords.shape)

# 绘制相机中心点到损伤点的连线
line_length = 10

# 计算方向向量并归一化
directions = defects_area_mask_world - chosen_camera_center_coords
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
# 延长后的终点
endpoints = defects_area_mask_world + directions * line_length

# 连接点云与相机中心点(第一行为相机中心点，其他为损伤点)
concate_camera_center_to_defects_points = np.vstack([chosen_camera_center_coords, endpoints])
print('############ concate_camera_center_to_defects_points ############', concate_camera_center_to_defects_points, concate_camera_center_to_defects_points.shape)


line_set = o3d.geometry.LineSet(
    points= o3d.utility.Vector3dVector(concate_camera_center_to_defects_points),
    lines = o3d.utility.Vector2iVector(np.array([[0, i+1] for i in range(len(concate_camera_center_to_defects_points)-1)]))
)
# line_set.points = o3d.utility.Vector3dVector(concate_camera_center_to_defects_points)
# line_set.lines = o3d.utility.Vector2iVector(np.array([[0, i+1] for i in range(0, concate_camera_center_to_defects_points.shape[0])]))
line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 1]] * (concate_camera_center_to_defects_points.shape[0]-1)))
# o3d.visualization.draw_geometries([line_set])

################################### 5. 射线与mesh的交点 #############################################
# 5.1 转换为 TensorMesh 并添加到 RaycastingScene
# 网格
mesh_dir = f'{render_mesh_dir}/train/ours_30000/fuse_post.ply'
# mesh_dir = None
mesh = o3d.io.read_triangle_mesh(mesh_dir) 
print('mesh:', mesh)
t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(t_mesh)

# 5.2 构建 rays tensor [N, 6]
chosen_camera_center_coords_same_shape_with_rays = np.tile(chosen_camera_center_coords, (directions.shape[0], 1))
rays = np.concatenate([chosen_camera_center_coords_same_shape_with_rays, directions], axis=1)
rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

# 5.3. 进行射线投射
ans = scene.cast_rays(rays_o3d)
t_hits = ans['t_hit'].numpy()
print('ans.keys():', ans.keys())
ray_hit_mesh = [mesh]
print('ray hit mesh的交点的shape:', t_hits.shape, t_hits)

################################### 6. open3d 显示 #############################################
# np_rand_dot = np.random.randn(100, 3)

# 计算空间点的距离
def distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)


# points
if mesh_dir is None and camera_center_coords is None:
    scene_points = np.vstack((scene_points, world_coords))
    pcd.points = o3d.utility.Vector3dVector(scene_points)

    # np_rand_dot_color = np.array([[1, 0, 0]] * world_coords.shape[0])
    # np_rand_dot_color = img_np_reshape
    # 添加原始点云的颜色
    # colors = scene_colors
    scene_colors = np.vstack((scene_colors, img_np_reshape))
    pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    vizualizer.add_geometry(pcd)
    vizualizer.add_geometry(coor)
    
    # 设置可视化选项
    opt = vizualizer.get_render_option()
    # opt.show_coordinate_frame = True  # 显示坐标系
    # opt.background_color = np.asarray([0, 0, 0])  # 设置背景颜色为黑色

    # 设置点云中点的大小
    opt.point_size = 1.0  # 可以根据需要调整点的大小
    
    vizualizer.run()
    
# mesh + points + 相机中心点
elif mesh_dir is not None and camera_center_coords is not None:


  
    
    ################# ray与mesh的交点 #################
    # 7. 可视化每条射线及交点
    vizualizer.add_geometry(mesh)
    hit_points = []
    hit_points_colors = []
    for i in range(len(directions)):
        if np.isfinite(t_hits[i]):
            hit_point = chosen_camera_center_coords[0] + directions[i] * t_hits[i]
            hit_points.append(hit_point)
            hit_points_colors.append(crack_mask_magma_np_color_not_zero_list[i])
            # print(f"hit point {hit_point}, {hit_point.shape}")

            # 绘制射线
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([chosen_camera_center_coords[0], hit_point]),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            line.paint_uniform_color([0, 1, 0])  # 绿色
            vizualizer.add_geometry(line)

            ## 绘制交点（红色小球）
            # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            # sphere.translate(hit_point)
            # sphere.paint_uniform_color([1, 0, 0])
            # ray_hit_mesh.append(sphere)
            
            ## 绘制交点 （红色的点）
            points = o3d.geometry.PointCloud()
            points.points = o3d.utility.Vector3dVector([hit_point])
            # points.paint_uniform_color([1, 0, 0])
            points.colors = o3d.utility.Vector3dVector([crack_mask_magma_np_color_not_zero_list[i]])
            vizualizer.add_geometry(points)
    ##################################################
    camera_center_points = camera_center_coords
    scene_points = np.vstack((scene_points, world_coords, camera_center_points, hit_points))
    pcd.points = o3d.utility.Vector3dVector(scene_points)
    print('Number of hit_points:', len(hit_points))
    marker_list = hit_points[-4:]
    p1 = marker_list[0]
    p2 = marker_list[1]
    p3 = marker_list[2]
    p4 = marker_list[3]
    
    dist1 = distance(p1, p2)
    dist2 = distance(p2, p3)
    dist3 = distance(p3, p4)
    dist4 = distance(p4, p1)
    
    mmp3dp = 132.0 / dist1
    print(f'dist1: {dist1}, dist2: {dist2}, dist3: {dist3}, dist4: {dist4}')
    print(f'真实尺寸 / 3D点云尺寸： 16.5x8 / dist1 = 132.0 / {dist1} = {mmp3dp} mmp3dp(mm per 3D point)')
    
    # 按照类别对hitpoints存储
    cls_hitpoints_dict = {}
    cls_hitpoints_color_dict = {}
    for cls in range(1, num_classes+1):
        

        try:
            cls_hitpoints_dict[cls] = [[] for i in range(len(defect_cls_dict[cls]))]
            cls_hitpoints_color_dict[cls] = [[] for i in range(len(defect_cls_dict[cls]))]
            # cls_hitpoints_dict[cls] = []
            # cls_hitpoints_color_dict[cls] = []
            
            cls_list = defect_cls_dict[cls]
            for i, cls_item in enumerate(cls_list):
                # cls_hitpoints_color_dict[cls][i] = []
                for points_id in range(len(cls_item)):
                    cls_hitpoints_dict[cls][i].append(hit_points[points_id])
                    cls_hitpoints_color_dict
            
        except Exception as e:
            # print('出错内容：', e)
            print(f"No crack edge found for class={e} in img_name={img_name}")
            continue
    # 存储hit_points
    pcd_color_dict = {'all_points_list': hit_points,
                  'all_colors_list' : crack_mask_magma_np_color_not_zero_list,
                  'only_defects_cls_length': cls_num_of_points_and_colors_dict,
                  'each_cls_num_points': cls_hitpoints_dict,
                  'each_cls_colors': cls_hitpoints_color_dict,
                  'mmp3dp': mmp3dp}

    npy_save_dir = os.path.join(f'{save_output_mesh_ray_dir}', f'hit_points_from_2d_npy_with_markers')
    os.makedirs(npy_save_dir, exist_ok=True)
    np.save(os.path.join(f'{npy_save_dir}', f'{img_name}.npy'), pcd_color_dict )
    # np_rand_dot_color = np.array([[1, 0, 0]] * world_coords.shape[0])
    # np_rand_dot_color = img_np_reshape
    # 添加原始点云的颜色
    # colors = scene_colors
    camera_center_coords_colors = np.array([[1, 0, 1]] * camera_center_points.shape[0])
    added_colors = np.array(hit_points_colors)
    # added_colors = img_np_reshape
    scene_colors = np.vstack((scene_colors, img_np_reshape, camera_center_coords_colors, added_colors))
    pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    
    vizualizer.add_geometry(pcd)
    # vizualizer.add_geometry(line_set)
    # o3d.visualization.draw_geometries([line_set])
    # vizualizer.add_geometry(mesh)
    
    # ray_hit_mesh = o3d.geometry.TriangleMesh(ray_hit_mesh)
    # 增加坐标
    # o3d.visualization.draw_geometries(scene_points)
    # o3d.visualization.draw_geometries(coor)
    # o3d.visualization.draw_geometries(ray_hit_mesh)
    # vizualizer.add_geometry(ray_hit_mesh)
    # vizualizer.add_geometry(sphere_projection)
    
    vizualizer.add_geometry(coor)
    
    
    
    
    # 设置可视化选项
    opt = vizualizer.get_render_option()
    # opt.show_coordinate_frame = True  # 显示坐标系
    opt.background_color = np.asarray([0, 0, 0])  # 设置背景颜色为黑色

    # 设置点云中点的大小
    opt.point_size = 3.0  # 可以根据需要调整点的大小
    # 保存可视化结果
    scene_save_dir = os.path.join(f'{save_output_mesh_ray_dir}', 'ray_hit_only_pcd_with_markers')
    os.makedirs(scene_save_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(f'{scene_save_dir }', f'{img_name}.ply'), pcd)
    print(f'已保存可视化结果: {scene_save_dir }/{img_name}.ply')
    
    # 将pcd，coor，ray_hit_mesh合并为一个可视化结果
    original_mesh_points = np.asarray(mesh.vertices)
    original_mesh_colors = np.asarray(mesh.vertex_colors)
    # 合并mesh和新增点和颜色
    combined_mesh_and_added_points = np.vstack((original_mesh_points, world_coords, camera_center_points, hit_points))
    combined_mesh_and_added_colors = np.vstack((original_mesh_colors, img_np_reshape, camera_center_coords_colors, added_colors))
    
    mesh.vertices = o3d.utility.Vector3dVector(combined_mesh_and_added_points)
    mesh.vertex_colors = o3d.utility.Vector3dVector(combined_mesh_and_added_colors)
    # 保存mesh
    
    ray_hit_mesh_save_dir = os.path.join(f'{save_output_mesh_ray_dir}', 'ray_hit_mesh_with_markers')
    os.makedirs(ray_hit_mesh_save_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(f'{ray_hit_mesh_save_dir}', f'{img_name}.ply'), mesh)
    print(f'已保存可视化结果: {ray_hit_mesh_save_dir}/{img_name}.ply')
    # o3d.io.write_point_cloud(os.path.join(f'{save_output_mesh_ray_dir}', 'pcd.ply'), pcd)
    
    vizualizer.run()  

# mesh + points
elif mesh_dir is not None and camera_center_coords is None:
    # 点
    # scene_points = world_coords
    scene_points = np.vstack((scene_points, world_coords))
    pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_colors = np.vstack((scene_colors, img_np_reshape))
    pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    # 网格
    mesh = o3d.io.read_triangle_mesh(mesh_dir)    
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    vizualizer.add_geometry(pcd)
    vizualizer.add_geometry(mesh)
    # vizualizer.add_geometry(pcd)
    vizualizer.add_geometry(coor)
    
        # 设置可视化选项
    opt = vizualizer.get_render_option()
    # opt.show_coordinate_frame = True  # 显示坐标系
    # opt.background_color = np.asarray([0, 0, 0])  # 设置背景颜色为黑色

    # 设置点云中点的大小
    opt.point_size = 1.0  # 可以根据需要调整点的大小
    
    vizualizer.run()
# 存储点云
# o3d.io.write_point_cloud('scene_added_area.ply', pcd)
vizualizer.destroy_window()