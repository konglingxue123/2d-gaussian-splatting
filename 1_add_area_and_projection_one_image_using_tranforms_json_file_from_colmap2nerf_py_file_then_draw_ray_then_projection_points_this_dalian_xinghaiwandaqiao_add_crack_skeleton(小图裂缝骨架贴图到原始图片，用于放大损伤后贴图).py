import json
from PIL import Image
import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw
import cv2
from matplotlib import pyplot as plt

img_name = '0001'
colmap_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/dalian_xinghaiwandaqiao_video_input_rgb'
render_mesh_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/dalian_xinghaiwandaqiao_rgb'
save_output_mesh_ray_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/dalian_xinghaiwandaqiao_rgb'


#########################  贴图裂缝骨架   ###################################

dilate_size = 2
add_xy1 = [900, 600]
# add_xy2 = [650, 650]

img_dir = f'{colmap_dir}/images/{img_name}.jpg'
# 在图片上绘制一块矩形红色区域
im = Image.open(img_dir)
im_np = np.array(im)


def paste_mask_on_image_colormap_add_same_size(im_np, crack_mask_red_np, add_xy1):
    mask_zero = np.zeros_like(im_np)
    # 获取图像和掩码的形状
    image_height, image_width = im_np.shape[:2]
    mask_height, mask_width = crack_mask_red_np.shape[:2]

    # 提取粘贴起始坐标
    x1, y1 = add_xy1

    # 检查粘贴位置是否越界
    if x1 < 0 or y1 < 0 or x1 + mask_width > image_width or y1 + mask_height > image_height:
        raise ValueError("粘贴位置越界，请调整位置或掩码大小。")
    # crack_mask_red_np数值小于200的位置为0，大于200的位置为255
    # crack_mask_red_np[crack_mask_red_np<200] = 0
    # crack_mask_red_np[crack_mask_red_np>=200] = 255
    # crack_mask_red_np 
    
    # 找到掩码中值为 1 的位置
    mask = crack_mask_red_np[..., 0] !=0
    # mask = crack_mask_red_np!= 0

    # 在mask_zero的指定位置更新颜色
    mask_zero[y1:y1 + mask_height, x1:x1 + mask_width][mask] = crack_mask_red_np[mask]
    mask_zero_id = mask_zero[..., 0] !=0
    
    mask_zero_pil = Image.fromarray(mask_zero)
    mask_zero_pil_l = mask_zero_pil.convert('L')
    mask_zero_np_l = np.array(mask_zero_pil_l)
    # 在图像的指定位置更新颜色
    im_np[0:0 + image_height, 0:0 + image_width][mask_zero_id] = mask_zero[mask_zero_id]

    return im_np, mask_zero_pil, mask_zero_np_l, mask_zero_pil_l

# 贴图裂缝
crack_mask_magma_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/projection/CFD_044_dist_on_skel_lee_pil_for_projection_magma.png'
# crack_mask_magma_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/output/projection/CFD_044.jpg'
crack_mask_magma_pil_l = Image.open(crack_mask_magma_dir)
crack_mask_magma_pil_l_np = np.array(crack_mask_magma_pil_l)
print(crack_mask_magma_pil_l_np.shape, np.unique(crack_mask_magma_pil_l_np))

crack_mask_magma_pil = crack_mask_magma_pil_l.convert('RGB')
crack_mask_magma_np = np.array(crack_mask_magma_pil)



_, _, mask_zero_same_size, mask_zero_same_size_pil_l = paste_mask_on_image_colormap_add_same_size(im_np, crack_mask_magma_np, add_xy1)
print('mask_zero_same_size.shape', mask_zero_same_size.shape) # h, w = (1080, 1920)
# non_zero_position = np.where(mask_zero_same_size !=0)
# print('non_zero_position', non_zero_position) # (array([ 600,  601,  602, ..., 1877, 1878, 1879]), array([400, 401, 402, ..., 797, 798, 799]))

crack_mask_magma_np_dilate2 = cv2.dilate(crack_mask_magma_np, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
# crack_mask_magma_np_dilate2 = crack_mask_magma_np
crack_mask_magma_np_dilate2_pil = Image.fromarray(crack_mask_magma_np_dilate2)

# result_image_magma = paste_mask_on_image_colormap(im_np, crack_mask_magma_np_dilate2, add_xy1)
result_image_magma, mask_zero_same_size_rgb_pil, _, _ = paste_mask_on_image_colormap_add_same_size(im_np, crack_mask_magma_np_dilate2, add_xy1)
result_image_magma_pil = Image.fromarray(result_image_magma)

h, w = mask_zero_same_size.shape
print('h, w', h, w)

plt.subplot(1, 2, 1)
plt.imshow(result_image_magma_pil)

plt.subplot(1, 2, 2)
plt.imshow(mask_zero_same_size_rgb_pil)
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
crack_mask_magma_pil_l_np_nonzero_id = np.where(mask_zero_same_size !=0)
print(crack_mask_magma_pil_l_np_nonzero_id[0].shape, crack_mask_magma_pil_l_np_nonzero_id[1].shape)
# 组合为二维坐标点，并调整形状为 [30000, 2]
x_defects_in_img = crack_mask_magma_pil_l_np_nonzero_id[1]
y_defects_in_img = crack_mask_magma_pil_l_np_nonzero_id[0]
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
print('maskzero_rgb_np.shape:', mask_zero_same_size_rgb_np)
defects_area_mask_world_list = []
crack_mask_magma_np_color_not_zero_list = []
# print('im.size, im.size[0],im.size[1]:', im.size, im.size[0],im.size[1])
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
ray_hit_mesh = [mesh]
print('ray hit mesh的交点的shape:', t_hits.shape, t_hits)

################################### 6. open3d 显示 #############################################
# np_rand_dot = np.random.randn(100, 3)



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
    o3d.io.write_point_cloud(os.path.join(f'{save_output_mesh_ray_dir}', 'scene_added_area_skeleton.ply'), pcd)
    print(f'已保存可视化结果: {save_output_mesh_ray_dir}/scene_added_area_skeleton.ply')
    
    # 将pcd，coor，ray_hit_mesh合并为一个可视化结果
    original_mesh_points = np.asarray(mesh.vertices)
    original_mesh_colors = np.asarray(mesh.vertex_colors)
    # 合并mesh和新增点和颜色
    combined_mesh_and_added_points = np.vstack((original_mesh_points, world_coords, camera_center_points, hit_points))
    combined_mesh_and_added_colors = np.vstack((original_mesh_colors, img_np_reshape, camera_center_coords_colors, added_colors))
    
    mesh.vertices = o3d.utility.Vector3dVector(combined_mesh_and_added_points)
    mesh.vertex_colors = o3d.utility.Vector3dVector(combined_mesh_and_added_colors)
    # 保存mesh
    
    
    o3d.io.write_triangle_mesh(os.path.join(f'{save_output_mesh_ray_dir}', 'ray_hit_mesh_skeleton.ply'), mesh)
    print(f'已保存可视化结果: {save_output_mesh_ray_dir}/ray_hit_mesh_skeleton.ply')
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