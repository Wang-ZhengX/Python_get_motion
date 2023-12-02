from project3_fengzhuang import *

# 打开视频文件
cap = cv2.VideoCapture('video/motion.avi')
# 读取视频第一帧frame0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame0 = cap.read()

# 读取视频cap中的帧率fps,总帧数frames
fps, frames = capget(cap)

# 框选，输出框选数据rois，框选数n
rois, n = templateget(frame0)

# 关闭窗口
cv2.destroyAllWindows()

# 重新读取第一帧作为原图
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
ret, frame0 = cap.read()

# 主函数，得到位移数据集datas
datas = main(frames,n,rois,frame0,cap)

print(datas)