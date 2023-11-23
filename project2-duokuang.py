import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard

# 打开视频文件
cap = cv2.VideoCapture('video/motion.avi')
# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)


# 定义位移结果的空列表
x_displacement = []
y_displacement = []
rois=[]

# 读取第一帧作为原图像
cap.set(cv2.CAP_PROP_POS_FRAMES,1)
ret, frame0 = cap.read()
# ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

n=-1
while True:
    # 框定特征点
    roi = cv2.selectROI(frame0)
    rois.append(roi)
    cv2.rectangle(frame0, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
    n = n + 1
    if keyboard.is_pressed('enter'):
        break

cv2.destroyAllWindows()

cap.set(cv2.CAP_PROP_POS_FRAMES,1)
ret, frame0 = cap.read()

# 循环读取视频帧
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break
        # if not ret判断ret的值是否为False，即判断是否读取失败。
        # 如果读取失败，表示视频的所有帧都已经读取完毕或发生了错误，break语句跳出循环

    i = 0
    while (i<n):
        x0, y0, w1, h1 = rois[i]  # 读取矩形框位置信息
        frame_roi = frame0[y0:y0 + h1, x0:x0 + w1]  # 将框内所有坐标选为模板
        yf = 100  # 竖向匹配范围值
        frame_fanwei = frame[y0 - yf:y0 + h1 + yf, x0:x0 + w1]  # 设置竖向匹配范围

        # cv2.matchTemplate()将frame_roi置于图像frame中匹配，得出每个位置的匹配度
        result = cv2.matchTemplate(frame_fanwei, frame_roi, cv2.TM_CCOEFF_NORMED)
        #TM_SQDIFF 平方差匹配法、TM_SQDIFF_NORMED 归一化平方差匹配法、TM_CCORR 相关匹配法
        #TM_CCORR_NORMED 归一化相关匹配法、TM_CCOEFF 系数匹配法、TM_CCOEFF_NORMED 归一化相关系数匹配法
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # 找出匹配度最大的框的左上角坐标
        # 最小匹配度、最大匹配度、最小匹配度坐标、最大匹配度坐标
        # 对于单通道图像，这些位置是二维坐标(x, y)；对于多通道图像，这些位置是三维坐标(x, y, c)，其中c表示通道索引

        # 计算每一帧特征值的相对位移
        x1 = max_loc[0]
        y1 = max_loc[1]
        dy = y1 - yf
        # 将所得数据放入列表
        y_displacement.append(dy)

        # 绘制每一帧与特征值匹配度最大矩形框
        cv2.rectangle(frame_fanwei, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.rectangle(frame, (x0, y0-yf), (x0+w1, y0+h1+yf), (255, 0, 0), 2)
        # 在矩形框上显示位移
        cv2.putText(frame, f"dy: {dy}", (x0, y0 - yf - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        i = i + 1

    #建立窗口并刷新图像
    cv2.imshow('Frame', frame)
    if cv2.waitKey(70) == ord('q'):
        break

cap.release() #释放硬件资源防止报错
cv2.destroyAllWindows()#关闭窗口

# 绘制位移随时间变化的图像
time = np.arange(len(x_displacement)) / fps

# 对位移时程进行FFT变换
fft_y = np.fft.fft(y_displacement)
# 获取频率轴
freq = np.fft.fftfreq(len(x_displacement), d=1/fps)
# 获取主要频率成分的索引

# 获取主要频率
main_freq_idx_y = np.argmax(np.abs(fft_y))
main_freq_y = freq[main_freq_idx_y]
print(f"Main frequency in y direction: {main_freq_y} Hz")

# 获取FFT结果的幅值谱
amplitude_y = np.abs(fft_y)

# 位移时程图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(time, y_displacement, label='y displacement')
plt.xlabel('t(s)')
plt.ylabel('displacement(pixel)')
plt.legend()

# 绘制结构幅值随频率变化的图像
plt.subplot(1, 2, 2)
plt.plot(freq[:len(freq)//2], amplitude_y[:len(freq)//2], label='y Amplitude')
plt.xlabel('frequency(Hz)')
plt.ylabel('A(pixel)')   # get Amplitude 获取振幅
plt.legend()

plt.tight_layout()
plt.show()