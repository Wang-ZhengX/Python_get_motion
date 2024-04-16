# Python_get_motion
**项目依赖：**
> pip install -r requirements.txt
___
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

---
## Update
## 2024/2/22
- 函数封装 
  - def GetFps(video_path)  -->获取视频帧数和视频文件
  - def GetTemplate(cap)   -->获取匹配模板信息(位置、像素矩阵)
  - def GetTemplateArea(VideoRoi, frame, yf)  -->获取逐帧匹配范围
  - def Getdy(template_area, VideoTemplate, yf)  -->获取y方向的位移
  - def Getfft(fps, y_displacement)  -->对位移时程作傅里叶变换
  - def ploty(time, y_displacement, freq, amplitude_y)  -->绘图处理

## 2/25
- SIFT算法进行两张图片特征点匹配(尚未运用到motion.py)  -->KSIFT.py

## 2/26
- 双线程处理数据-->pyqueue.py

## 3/3
- 框选多个感兴趣区域进行位移提取-->motion_select.py&pyselect.py

## 3/9
- 位移提取结合KSIFT.py

## 03/16   形态学处理
- 腐蚀（erode）:消除边界点，使边界向内部收缩，用来消除小且无意义的物体
- 膨胀(dilate)：“加长”或“变粗”二值图像中的对象，减小RGB值为零的孔洞（黑色区域）
- 开运算（open）：先腐蚀后膨胀，平滑对象轮廓，断开狭窄的连接，消除细的突出物
- 闭运算(close)：先膨胀后腐蚀，可填补狭窄的缺口，填充小洞
