import cv2
import numpy as np
import matplotlib.pyplot as plt
from KSIFT import getsift


def GetFps(video_path):
    # 读取视频帧和视频文件
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps


def GetTemplate(cap):
    # 获取模板，返回模板位置信息和像素矩阵
    cap.set(cv2.CAP_PROP_POS_FRAMES, 800)
    _, frame0 = cap.read()
    # frame0 = cv2.resize(frame0, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    cv2.namedWindow('frame0', cv2.WINDOW_NORMAL)
    roi0 = cv2.selectROI('frame0',frame0)
    x0, y0, w1, h1 = roi0
    frame_roi = frame0[y0: y0 + h1, x0: x0 + w1]
    cv2.destroyAllWindows()  # 关闭窗口
    return roi0, frame_roi


def GetTemplateArea(VideoRoi, frame, yf = 100):
    # 获取逐帧匹配的范围
    x0, y0, w1, h1 = VideoRoi
    frame_Area = frame[y0 - yf: y0 + h1 + yf, x0: x0 + w1]
    return frame_Area, yf


def Getdy(template_area, VideoTemplate, h1, w1, yf = 100):
    # 获取y方向的位移
    result = cv2.matchTemplate(template_area, VideoTemplate, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    x1 = max_loc[0]
    y1 = max_loc[1]
    frame_template = template_area[y1: y1 + h1, x1: x1 + w1]
    keypoints_1, keypoints_2, src_pts, dst_pts, good = getsift(frame_template, VideoTemplate)
    sift_displacement = src_pts - dst_pts
    # if判断sift_displacement是否为空列表
    if sift_displacement.any():
        # dylist = np.take(displacement, 1, axis=1)
        dylist = sift_displacement[:, 1]
        sift_dy = np.around(np.mean(dylist),2)
        dy = y1 - yf + sift_dy

    else:
        dy = y1 - yf
    return x1, y1, dy


def Getfft(fps, y_displacement):
    # 绘制位移随时间变化的图像
    time = np.arange(len(y_displacement)) / fps

    # 对位移时程进行FFT变换
    fft_y = np.fft.fft(y_displacement)
    # 获取频率轴
    freq = np.fft.fftfreq(len(y_displacement), d=1 / fps)
    # 获取主要频率成分的索引

    # 获取主要频率
    main_freq_idx_y = np.argmax(np.abs(fft_y))
    main_freq_y = freq[main_freq_idx_y]
    print(f"Main frequency in y direction: {'%.5f' % main_freq_y} Hz")

    # 获取FFT结果的幅值谱
    amplitude_y = np.abs(fft_y)

    return time, y_displacement, freq, amplitude_y


def ploty(time, y_displacement, freq, amplitude_y):
    # 位移时程图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time, y_displacement, label='y displacement', linewidth=1)
    plt.xlabel('t(s)')
    plt.ylabel('displacement(pixel)')
    plt.legend()

    # 绘制结构幅值随频率变化的图像
    plt.subplot(1, 2, 2)
    plt.plot(freq[:len(freq) // 2], amplitude_y[:len(freq) // 2], label='y Amplitude', linewidth=1)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('A(pixel)')  # get Amplitude 获取振幅
    plt.legend()
    plt.tight_layout()
    plt.show()
    return plt.show()



if __name__ == '__main__':

    video = 'video/motion.avi'
    y_displacement = []
    cap, fps = GetFps(video)
    VideoRoi, VideoTemplate = GetTemplate(cap)
    x0, y0, w1, h1 = VideoRoi

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    # 循环读取视频帧
    while cap.isOpened():
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        template_area, yf = GetTemplateArea(VideoRoi, frame)

        x1, y1, dy = Getdy(template_area, VideoTemplate, h1, w1)

        # 将所得数据放入列表
        y_displacement.append(dy)

        # 绘制每一帧与特征值匹配度最大矩形框
        cv2.rectangle(template_area, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
        cv2.rectangle(frame, (x0, y0 - yf), (x0 + w1, y0 + h1 + yf), (0, 255, 0), 2)
        # 在矩形框上显示位移
        cv2.putText(frame, f"dy: {dy}", (x0, y0 - yf - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # 建立窗口并刷新图像
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()  # 释放硬件资源防止报错
    cv2.destroyAllWindows()  # 关闭窗口

    time, y_displacement, freq, amplitude_y = Getfft(fps, y_displacement)
    a = ploty(time, y_displacement, freq, amplitude_y)