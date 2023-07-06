import cv2
import numpy as np

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'鼠标坐标:({x},{y})')

# 创建一个空白的窗口，并设置回调函数
cv2.namedWindow('image')
cv2.setMouseCallback('image', on_mouse_click)

# 在窗口中显示一张图片
img = np.zeros((300, 300, 3), np.uint8)
while(True):
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放窗口和资源
cv2.destroyAllWindows()