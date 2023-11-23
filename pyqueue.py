from pymotion import *
import queue
import time
import random
import threading

video = 'video/motion.avi'
cap, fps = GetFps(video)
VideoRoi, VideoTemplate = GetTemplate(cap)
x0, y0, w1, h1 = VideoRoi


def tempalte_dy(q):
    # 匹配获取dy线程
    while cap.isOpened():
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        template_area, yf = GetTemplateArea(VideoRoi, frame)

        x1, y1, dy = Getdy(template_area, VideoTemplate)
        q.put(dy)

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



def tempalte_plot(q):
    # 绘图线程
    while True:
        data = q.get()  # 从队列中获取数据
        print(f"dy {data}")
        q.task_done()  # 标记任务完成
        # time.sleep(0.02)

if __name__ == '__main__':
    q = queue.Queue()  # 创建一个队列对象

    # 创建生产者线程和消费者线程
    producer_thread = threading.Thread(target=tempalte_dy, args=(q,))
    consumer_thread = threading.Thread(target=tempalte_plot, args=(q,))

    # 启动线程
    producer_thread.start()
    consumer_thread.start()

    # 等待线程结束
    producer_thread.join()
    consumer_thread.join()