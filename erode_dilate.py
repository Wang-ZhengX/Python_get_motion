import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
image = cv2.imread('image/41.jpg', cv2.IMREAD_GRAYSCALE)

# 定义结构元素
kernel = np.ones((5, 5), np.uint8)

# 膨胀操作
dilated_image = cv2.dilate(image, kernel, iterations=1)

# 腐蚀操作
eroded_image = cv2.erode(image, kernel, iterations=1)

# 开运算
opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 闭运算
closing_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 显示处理后的图片
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].imshow(dilated_image, cmap='gray')
axs[0, 0].set_title('Dilated Image')

axs[0, 1].imshow(eroded_image, cmap='gray')
axs[0, 1].set_title('Eroded Image')

axs[1, 0].imshow(opening_image, cmap='gray')
axs[1, 0].set_title('Opening Image')

axs[1, 1].imshow(closing_image, cmap='gray')
axs[1, 1].set_title('Closing Image')

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
