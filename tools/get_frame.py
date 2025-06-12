import cv2
import os
def frame_cap(video_dir, save_img_dir, frame_interval):
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    vc = cv2.VideoCapture(video_dir)  # 读入视频文件，命名cv
    n = 1  # 计数

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    timeF = frame_interval  # 视频帧计数间隔频率

    i = 0
    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            i += 1
            print(i)
            cv2.imwrite(f'{save_img_dir}/{i}.jpg', frame)  # 存储为图像
        n = n + 1
        # cv2.waitKey(1)
    vc.release()

if __name__ == '__main__':
    # video_dir = r'C:\Users\kong\Desktop\doc\barrel\video\VID_20230506_100206.mp4'
    video_dir = '/home/ubunto/Project/konglx/pcd/3dgs/gaussian-splatting-main/datasets/北京昌平-百葛桥-钢管混凝土腐蚀/WeChat_20250507133211.mp4'
    # save_img_dir = r'C:\Users\kong\Desktop\doc\barrel\framesplit'
    save_img_dir = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/beijing_baigeqiao_1'
    frame_interval = 25
    frame_cap(video_dir, save_img_dir, frame_interval)
