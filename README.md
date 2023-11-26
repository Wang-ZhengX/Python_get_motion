# Python_get_motion

## 1.motion.py
调用pymotion.py中的自定义函数

## 2.队列(Queue)-->pyqueue.py
作用：缓冲

queue.Queue	先进先出队列
- FIFO (first in first out 先进先出)
- 可以储存不同的数据类型，例如整数、字符串、字典
- 使用put存入数据，使用get取数据(如果当前队列中没有数据，则此时会堵塞)

queue.LifoQueue	后进先出队列(堆栈)
- LIFO(last in first out 后进先出)
- 可以储存不同的数据类型，例如整数、字符串、字典
- 使用put存入数据，使用get取数据(如果当前队列中没有数据，则此时会堵塞)

queue.PriorityQueue	优先级队列
- 根据优先级来确定当前要获取的数据
- 使用put存放
  - 将一个元组放入
  - 第1个元素是：优先级，数字越小优先级越高
  - 第2个元素是：要存放的数据
- 使用get取数据

## 2023/11/22
- 函数封装 
  - def GetFps(video_path)  -->获取视频帧数和视频文件
  - def GetTemplate(cap)   -->获取匹配模板信息(位置、像素矩阵)
  - def GetTemplateArea(VideoRoi, frame, yf)  -->获取逐帧匹配范围
  - def Getdy(template_area, VideoTemplate, yf)  -->获取y方向的位移
  - def Getfft(fps, y_displacement)  -->对位移时程作傅里叶变换
  - def ploty(time, y_displacement, freq, amplitude_y)  -->绘图处理

## 11/25
- SIFT算法进行两张图片特征点匹配(尚未运用到motion.py)  -->KSIFT.py