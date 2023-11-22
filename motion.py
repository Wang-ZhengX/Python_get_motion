from pymotion import *

video = 'video/motion.avi'
x_displacement = []
y_displacement = []
cap, fps = GetFps(video)
VideoRoi, VideoTemplate = GetTemplate(cap)
x0, y0, w1, h1 = VideoRoi

# 循环读取视频帧
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break
    template_area, yf = TemplateArea(VideoRoi, frame, 100)
    result = cv2.matchTemplate(template_area, VideoTemplate, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 计算每一帧特征值的相对位移
    x1 = max_loc[0]
    y1 = max_loc[1]
    dy = y1 - yf + 23
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

a = plot(time, y_displacement, freq, amplitude_y)

