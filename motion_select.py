from pyselect import *
from pymotion import *

video = 'video/motion.avi'
# 读取视频cap中的帧率fps,第一帧frame0,总帧数frames
cap, fps, frames = capget(video)

frame0 = frameget(cap)
# 框选，输出框选数据rois，框选数n
rois, n = templateget(frame0)

framezero = frameget(cap)
# 主函数，得到位移数据集datas
datas = main(frames, n, rois, framezero, cap)

for i in range(n):
    time, y_displacement, freq, amplitude_y = Getfft(fps, datas[i].tolist())
    a = ploty(time, y_displacement, freq, amplitude_y)