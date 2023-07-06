# 中山大学深度学习大作业--yolov5+lpr3+deepsort交通识别检测系统

本次大作业，我们小组完成了基于[Yolov5](https://github.com/ultralytics/yolov5) 的车型、行人、车牌检测，基于[lpr3](https://github.com/szad670401/HyperLPR)的车牌号码识别以及基于[deepsort](https://github.com/dyh/unbox_yolov5_deepsort_counting)的车流量计数任务，并将所有功能集成为一体，完成了一个交互式的交通识别检测GUI系统。

![](https://github.com/donahowe/traffic-detect-GUI/blob/main/GUI_picture/window.jpg)

![](https://github.com/donahowe/traffic-detect-GUI/blob/main/GUI_picture/1.gif)

![](https://github.com/donahowe/traffic-detect-GUI/blob/main/GUI_picture/2.gif)

## Our Idea

This project is a faithful PyTorch implementation

## Installation

快速安装操作：
```
git clone https://github.com/donahowe/traffic-detect-GUI.git
cd traffic-detect-GUI
pip install -r requirements.txt
```

若要在本地电脑运行GUI系统，还需要安装[QT Designer](https://blog.csdn.net/qq_32892383/article/details/108867482)。请依据连接中给出的参考下载步骤安装。

## How To Run?

### Quick Start

运行我们的GUI界面：
```
python mymain.py
```
如果视频较长或视频中车流量较慢，在不使用GPU加速的情况下检测需要一定时间，待界面左下角提示 **Ready!** 时，便可点击 **开始** 查看结果。

### Pre-trained Models

您可以在[此处]([https://drive.google.com/drive/folders/1jIr8dkvefrQmv737fFm2isiT6tqpbTbv](https://drive.google.com/file/d/1qMw3ofK_nJauSrvDFTfoThprqvynLgB7/view?usp=sharing)https://drive.google.com/file/d/1qMw3ofK_nJauSrvDFTfoThprqvynLgB7/view?usp=sharing)下载我们的yolov5车型+行人检测预训练模型。
