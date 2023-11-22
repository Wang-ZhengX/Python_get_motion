from pymotion import *
import queue
import time
import random
import threading

video = 'video/motion.avi'
cap, fps = GetFps(video)
VideoRoi, VideoTemplate = GetTemplate(cap)
x0, y0, w1, h1 = VideoRoi

# 生产者线程
def producer(q):
    while cap.isOpened():
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        template_area, yf = TemplateArea(VideoRoi, frame, 100)
        result = cv2.matchTemplate(template_area, VideoTemplate, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 计算每一帧特征值的相对位移
        y1 = max_loc[1]
        dy = y1 - yf
        y_displacement = q.put(dy)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        time.sleep(0.01)



# 消费者线程
def consumer(q):
    while True:
        data = q.get()  # 从队列中获取数据
        print(f"Consumed {data}")
        q.task_done()  # 标记任务完成
        time.sleep(0.02)

if __name__ == '__main__':
    q = queue.Queue()  # 创建一个队列对象

    # 创建生产者线程和消费者线程
    producer_thread = threading.Thread(target=producer, args=(q,))
    consumer_thread = threading.Thread(target=consumer, args=(q,))

    # 启动线程
    producer_thread.start()
    consumer_thread.start()

    # 等待线程结束
    producer_thread.join()
    consumer_thread.join()