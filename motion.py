from pymotion import *

video = 'video/motion.avi'
x_displacement = []
y_displacement = []
cap, fps = GetFps(video)
VideoRoi, VideoTemplate = GetTemplate(cap)
x0, y0, w1, h1 = VideoRoi
y_f = 100


# 循环读取视频帧
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break
    template_area, yf = TemplateArea(VideoRoi, frame, y_f)

    x1, y1, dy = Getdy(template_area, VideoTemplate, y_f)

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

a = plot(time, y_displacement, freq, amplitude_y)

