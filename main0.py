import numpy as np
import cv2
import time
import tracker
from detector import Detector

def Count_num(road,send_raw,send_img,is_continue):

    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)

    # 初始化2个撞线polygon   蓝色
    # list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
    #                 [299, 375], [267, 289]]
    list_pts_blue = [[204, 305+200], [227, 431+200], [605, 522+200], [1101, 464+200], [1900, 601+200], [1902, 495+200], [1125, 379+200], [604, 437+200],
                     [299, 375+200], [267, 289+200]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon   黄色
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    # list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
    #                    [594, 637], [118, 483], [109, 303]]
    list_pts_yellow = [[181, 305+200], [207, 442+200], [603, 544+200], [1107, 485+200], [1898, 625+200], [1893, 701+200], [1101, 568+200],
                       [594, 637+200], [118, 483+200], [109, 303+200]]
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

    # 进入数量
    down_count = 0
    # 离开数量
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
    capture = cv2.VideoCapture(road)
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    # print(fps)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(width)
    # location 定义为存储前一帧和后一帧的列表
    location = []       # location[0] prev    location[1] after
    global speed_all
    global output_image_frame
    i = 0
    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break
        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))
        print(is_continue)
        while not is_continue:  # 只要 is_continue 变量为 False 就继续等待
            print("等待 5 秒钟后再次检查 is_continue 的值")
            time.sleep(5)  # 暂停 5 秒钟
            if is_continue:
                # 如果 is_continue 变量为 True，就跳出循环，执行程序代码
                break

        send_raw.emit(im)
        list_bboxs = []
        bboxes = detector.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 1:    # 用来存储每一个的信息
            list_bboxs = tracker.update(bboxes, im)
            if len(list_bboxs) > 0:
                list_bboxs_sp = list_bboxs
            print("bbox: ", bboxes)
            print("list_bbox: ", list_bboxs)
            if len(list_bboxs) > 0:
                location.append(list_bboxs_sp)
                speed_all = list(np.zeros(len(bboxes)))
                if len(location) == 2:
                    speed_all = tracker.Estimated_speed(location, fps=fps, width=960.0)
                    # speed_id = speed_all[:, 0]
                    print("speed_all[0]:  ", speed_all[0])

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                    output_image_frame = tracker.draw_bboxes(im, list_bboxs, speed=speed_all, line_thickness=None)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass
        if len(location) >= 2:
            location[0] = location[1]
            location.pop()

        # 输出图片
        if i == 0:
            output_image_frame = color_polygons_image
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)
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

                if polygon_mask_blue_and_yellow[y, x] == 1 and label not in ['person', 'bicycle', 'motorcycle']:
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
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=0.8, color=(0, 0, 0), thickness=2)

        # cv2.imshow('demo', output_image_frame)
        cv2.waitKey(1)
        # return output_image_frame
        send_img.emit(output_image_frame)
        # cv2.waitKey(1)
        pass
    pass

    capture.release()
    # cv2.destroyAllWindows()
    # return output_image_frame

if __name__ == '__main__':
    road = 'E:/images/test.mp4'
    Count_num(road)
    # cv2.imshow('demo', output_image)
    # cv2.waitKey(0)