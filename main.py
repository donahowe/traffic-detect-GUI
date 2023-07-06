import numpy as np
import tracker
from detector import Detector
import cv2

# 导入cv相关库及预处理函数 用于车牌识别
from show_license_plate import draw_plate_on_image
from PIL import ImageFont
# 导入依赖包
import hyperlpr3 as lpr3

global list_blue
global list_yellow
list_blue = []
list_yellow = []

def on_mouse_click_yellow(event, x, y, flags, param):         # 定义回调函数，用于获取鼠标位置
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'鼠标坐标:({x},{y})')
        list_yellow.append([x, y])                             # 用来存储需要标黄线的区域坐标，以便接下来标注区域

def on_mouse_click_blue(event, x, y, flags, param):         # 定义回调函数，用于获取鼠标位置
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'鼠标坐标:({x},{y})')
        list_blue.append([x, y])                              # 用来存储需要标黄线的区域坐标，以便接下来标注区域
def count_speed(path_video):                                 # 将实现车辆计数和车速识别代码封装成一个函数，path_video 表示数据集视频的路径
        capture_fir = cv2.VideoCapture(path_video)           # 读取数据集
        k = 0
        while True:
            k += 1   # K用来标记读取了第几帧图片，第一帧照片用来标定蓝线，第二帧照片用来标定黄线
            _, im = capture_fir.read()  # 读取帧照片
            if im is None:    # 当未读到视频帧时，则表示视频已经结束，则跳出读取的过程
                break
            if k == 1:  # 当读到第一帧照片时，标蓝区域
                cv2.namedWindow('image')  # 创建一个空白的窗口，并设置回调函数
                cv2.imshow('image', im)   # 在输出的图像中标定坐标
                cv2.setMouseCallback('image', on_mouse_click_blue)
                while (True):
                    if cv2.waitKey(1) & 0xFF == ord('q'):   # 无限循环，当键盘输入‘q’并回车时，结束读取坐标的过程，保证了对区域的可选择性较高
                        break
                cv2.destroyAllWindows()   # 释放窗口和资源
            if k == 2: # 当读到第一帧照片时，标黄区域
                cv2.namedWindow('image')   # 创建一个空白的窗口，并设置回调函数
                cv2.imshow('image', im)    # 在输出的图像中标定坐标
                cv2.setMouseCallback('image', on_mouse_click_yellow)
                while (True):
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # 无限循环，当键盘输入‘q’并回车时，结束读取坐标的过程，保证了对区域的可选择性较高
                        break
                cv2.destroyAllWindows()    # 释放窗口和资源
                break
        # 将蓝、黄 两区域的坐标输出，观察其准确性
        print("list_yellow: ", list_yellow)
        print("list_blue: ", list_blue)

        # 根据视频尺寸，填充一个polygon，供撞线计算使用
        mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)

        # 初始化2个撞线polygon   蓝色  此为模型手动输入蓝色区域的范围 ，对于测定来往车辆计数时较为便利
        # list_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
        #                 [299, 375], [267, 289]]
        # list_blue = [[204, 305+200], [227, 431+200], [605, 522+200], [1101, 464+200], [1900, 601+200], [1902, 495+200], [1125, 379+200], [604, 437+200],
        #                  [299, 375+200], [267, 289+200]]
        list_pts_blue = list_blue
        ndarray_pts_blue = np.array(list_pts_blue, np.int32)
        polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
        polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

        # 填充第二个polygon   黄色  此为模型手动输入黄色区域的范围 ，对于测定来往车辆计数时较为便利
        mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
        # list_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
        #                    [594, 637], [118, 483], [109, 303]]
        # list_yellow = [[181, 305+200], [207, 442+200], [603, 544+200], [1107, 485+200], [1898, 625+200], [1893, 701+200], [1101, 568+200],
        #                    [594, 637+200], [118, 483+200], [109, 303+200]]
        list_pts_yellow = list_yellow
        ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
        polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
        polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

        # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
        polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

        # 缩小尺寸，1920x1080->960x540
        polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

        # 蓝 色盘 b,g,r
        blue_color_plate = [255, 0, 0]
        # 蓝 polygon图片
        blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

        # 黄 色盘
        yellow_color_plate = [0, 255, 255]
        # 黄 polygon图片
        yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

        # 彩色图片（值范围 0-255）
        color_polygons_image = blue_image + yellow_image
        # 缩小尺寸，1920x1080->960x540
        color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

        # list 与蓝色polygon重叠
        list_overlapping_blue_polygon = []

        # list 与黄色polygon重叠
        list_overlapping_yellow_polygon = []

        # 进入车辆数量
        down_count = 0
        # 离开车辆数量
        up_count = 0
        # 统计car数量
        car_count = 0
        # 统计bus数量
        bus_count = 0
        # 统计truck数量
        truck_count = 0
        font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
        draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

        # 初始化 yolov5
        detector = Detector()

        # 打开视频
        capture = cv2.VideoCapture(path_video)
        # 读取视频帧数
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        # 读取视频帧图像的宽度
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

        # location 定义为存储前一帧和后一帧的列表，用来实现车速计算功能
        location = []       # location[0] prev    location[1] after
        global speed_all    # 定义全局变量，用来存储车速信息
        global output_image_frame  # 定义输出帧对象
        i = 0     # index
        # 对车牌识别进行函数的预处理
        # 中文字体加载
        font_ch = ImageFont.truetype("platech.ttf", 20, 0)
        # 实例化识别对象
        catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
        # 循环读取每一帧信息，并对此进行相关操作
        while True:
            # 读取每帧图片
            _, im = capture.read()
            if im is None:
                break

            # 缩小尺寸，1920x1080->960x540
            im = cv2.resize(im, (960, 540))

            list_bboxs = []  # 用来存储图像中出现的车或人的 label，id，图像像素中心 x和y的坐标
            bboxes = detector.detect(im)

            # 如果画面中 有bbox 即存在物体属于 label
            if len(bboxes) > 1:    # 用来存储每一个的信息
                list_bboxs = tracker.update(bboxes, im)  # 不断更新list_bboxs中的数据
                if len(list_bboxs) > 0:
                    list_bboxs_sp = list_bboxs
                print("bbox: ", bboxes)
                print("list_bbox: ", list_bboxs)
                if len(list_bboxs) > 0:
                    location.append(list_bboxs_sp)
                    speed_all = list(np.zeros(len(bboxes)))
                    if len(location) == 2:    # 当已经存储前后两帧图像信息时
                        speed_all = tracker.Estimated_speed(location, fps=fps, width=960.0)  # 由于已经改变图像尺寸，因此宽度为960.0

                # 画框
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                        output_image_frame = tracker.draw_bboxes(i, im, list_bboxs, speed=speed_all, line_thickness=None)
                pass
            else:
                # 如果画面中 没有bbox
                output_image_frame = im
            pass
            if len(location) >= 2:   # 当已经存储了前后两帧图像信息时，当要继续操作，则需要将location[0]信息删除
                location[0] = location[1]
                location.pop()

            # 输出图片
            if i == 0:
                output_image_frame = color_polygons_image
            output_image_frame = cv2.add(output_image_frame, color_polygons_image)   # 在图像中叠加处理好的信息

            # 对output_image_frame加入车牌识别
            # 执行识别算法
            results = catcher(output_image_frame)
            # print('车牌',results)
            for code, confidence, type_idx, box in results:
                # 解析数据并绘制
                text = f"{code} - {confidence:.2f}"
                output_image_frame = draw_plate_on_image(output_image_frame, box, text, font=font_ch)

            # 对output_image_frame加入车牌识别信息完成
            i += 1

            if len(list_bboxs) > 0:
                # ----------------------判断撞线----------------------
                for item_bbox in list_bboxs:
                    x1, y1, x2, y2, label, track_id = item_bbox

                    # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                    y1_offset = int(y1 + ((y2 - y1) * 0.6))

                    # 撞线的点
                    y = y1_offset
                    x = x1

                    if polygon_mask_blue_and_yellow[y, x] == 1 and label not in ['person', 'bicycle', 'motorcycle']:   # 行人、自行车、摩托车不在车辆计数的要求内
                        # 如果撞 蓝polygon
                        if track_id not in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.append(track_id)
                        pass

                        # 判断 黄polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 外出方向
                        if track_id in list_overlapping_yellow_polygon:
                            # 外出+1
                            up_count += 1
                            if label == 'bus':
                                bus_count += 1
                            elif label == 'car':
                                car_count += 1
                            else:
                                truck_count += 1

                            print(f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_yellow_polygon}')

                            # 删除 黄polygon list 中的此id
                            list_overlapping_yellow_polygon.remove(track_id)

                            pass
                        else:
                            # 无此 track_id，不做其他操作
                            pass

                    elif polygon_mask_blue_and_yellow[y, x] == 2 and label not in ['person', 'bicycle', 'motorcycle']:
                        # 如果撞 黄polygon
                        if track_id not in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.append(track_id)
                        pass

                        # 判断 蓝polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 进入方向
                        if track_id in list_overlapping_blue_polygon:
                            # 进入+1
                            down_count += 1
                            if label == 'bus':
                                bus_count += 1
                            elif label == 'car':
                                car_count += 1
                            else:
                                truck_count += 1
                            print(f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')

                            # 删除 蓝polygon list 中的此id
                            list_overlapping_blue_polygon.remove(track_id)

                            pass
                        else:
                            # 无此 track_id，不做其他操作
                            pass
                        pass
                    else:
                        pass
                    pass

                pass

                # ----------------------清除无用id----------------------
                list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
                for id1 in list_overlapping_all:
                    is_found = False
                    for _, _, _, _, _, bbox_id in list_bboxs:
                        if bbox_id == id1:
                            is_found = True
                            break
                        pass
                    pass

                    if not is_found:
                        # 如果没找到，删除id
                        if id1 in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.remove(id1)
                        pass
                        if id1 in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.remove(id1)
                        pass
                    pass
                list_overlapping_all.clear()
                pass

                # with open(r"./result/demo_" + str(i) + ".txt", 'w') as f:
                #     # for i in range(5):
                #     f.write('DOWN: ' + str(down_count) + '\n')
                #     f.write('UP: ' + str(up_count) + '\n')
                #     f.write('CAR: ' + str(car_count) + '\n')
                #     f.write('TRUCK: ' + str(truck_count) + '\n')
                #     f.write('BUS: ' + str(bus_count) + '\n')
                #     if results != []:
                #         f.write('车牌：' + results[0][0] + '\n')
                #         f.write('置信度：' + str(results[0][1]) + '\n')
                #     else:
                #         f.write('车牌：' + '0' + '\n')
                #         f.write('置信度：' + '0' + '\n')
            #     # 清空list
            #     list_bboxs.clear()
            #
            #     pass
            # else:
            #     # 如果图像中没有任何的bbox，则清空list
            #     list_overlapping_blue_polygon.clear()
            #     list_overlapping_yellow_polygon.clear()
            #     pass
            # pass

            text_draw = 'DOWN: ' + str(down_count) + \
                        ' , UP: ' + str(up_count) + \
                        ' , CAR: ' + str(car_count) + \
                        ' , TRUCK: ' + str(truck_count) + \
                        ' , BUS: ' + str(bus_count)


            with open(r"./result/demo_"+str(i)+".txt", 'w') as f:
                # for i in range(5):
                f.write('DOWN:\t' + str(down_count) + '\n')
                f.write('UP:\t' + str(up_count) + '\n')
                f.write('CAR:\t' + str(car_count) + '\n')
                f.write('TRUCK:\t' + str(truck_count) + '\n')
                f.write('BUS:\t' + str(bus_count) + '\n')
                # if results != []:
                #     f.write('车牌:\t' + results[0][0] + '\n')
                #     f.write('置信度:\t' + str(results[0][1]) + '\n')
                # else:
                #     f.write('车牌:\t' + '0' + '\n')
                #     f.write('置信度:\t' + '0' + '\n')
                # f.write('车牌：' + results[0][0] + '\n')
                # f.write('置信度：' + results[0][1] + '\n')
                # f.write('车辆总数：' + str() + ''

            output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                             org=draw_text_postion,
                                             fontFace=font_draw_number,
                                             fontScale=0.8, color=(0, 0, 0), thickness=2)   # 包含了图像、id、label、speed 等信息

            cv2.imwrite("./result/demo_"+str(i)+".jpg", output_image_frame)      # 将处理后的视频帧以图片形式保存下来到result文件夹中
            # cv2.imshow('demo', output_image_frame)
            cv2.waitKey(1)
            pass
        pass
        print("Down!")
        capture.release()
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    # list_blue = []
    # list_yellow = []

    count_speed('E:/images/test2.mp4')
