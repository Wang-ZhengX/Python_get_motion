import cv2
import matplotlib.pyplot as plt
import numpy as np

def getsift(img1, img2):

    # 获取参考图片尺寸，并依据参考图片对另一图片进行缩放
    reference_h, reference_w = img1.shape[:2]
    img2 = cv2.resize(img2, (reference_w, reference_h))

    # 创建SIFT特征检测器
    sift = cv2.xfeatures2d.SIFT_create()

    # 提取特征点和描述符
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # K-D树索引
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 100)  # checks键指定了FLANN搜索的参数，表示进行搜索时需要进行的检查次数

    # 创建了一个FLANN匹配器对象flann，并将之前定义的index_params和search_params作为参数传入，用于进行特征点匹配
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 进行K近邻匹配，即对每个特征点找到最近的两个匹配
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    # 从较为可靠的匹配点中提取出对应的特征点坐标，并将其存储在src_pts和dst_pts数组中
    # queryIdx和trainIdx分别表示对第一张图和第二张图中的特征点进行索引
    src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return keypoints_1, keypoints_2, src_pts, dst_pts, good

if __name__ == '__main__':
    img1 = cv2.imread('image/dot1.png', 0)
    img2 = cv2.imread('image/dot2.png', 0)
    keypoints_1, keypoints_2, src_pts, dst_pts, good = getsift(img1, img2)
    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good[:100], img2, flags=2)
    plt.imshow(img3)
    plt.show()