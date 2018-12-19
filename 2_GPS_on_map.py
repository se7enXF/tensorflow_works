# -*- coding:utf-8 -*-
'''
用途：
命令行参数：
输出：
'''

import numpy as np
import csv
from PIL import Image
import matplotlib.pylab as plt
import datetime
import os

x_max = 174     # 1~174（高）
y_max = 222     # 1~222（宽）

event_p = [[], []]
# [img_name, x1, y1, x2, y2, pos_x, pos_y]
img_x1 = [[0 for i in range(222)] for i in range(174)]
img_y1 = [[0 for i in range(222)] for i in range(174)]
img_x2 = [[0 for i in range(222)] for i in range(174)]
img_y2 = [[0 for i in range(222)] for i in range(174)]

string = [["大图x"], ["大图y"], ["图中x"], ["图中y"]]
map_string = [["图片名"], ["label"]]
map_gps_dir = "D:/data/GPS_map.csv"


def map_gps(train_label, NYPD):

    # 读入NYPD信息
    with open(NYPD, "r") as f11:
        c_reader = csv.reader(f11)
        csv_line_num = 0
        for item in c_reader:
            csv_line_num += 1
            if csv_line_num == 1:
                continue

            # 时间事故中心
            event_p[0].append(float(item[3]))
            event_p[1].append(float(item[2]))

    # 读取原始train图片信息
    print("读取所有train图片信息......")
    with open(train_label, "r") as File:
        c_reader = csv.reader(File)
        csv_line_num = 0
        for item in c_reader:
            csv_line_num += 1
            if csv_line_num == 1:
                continue

            # 读入图片信息
            [img_name, x1, y1, x2, y2, pos_x, pos_y] = item
            [pos_x, pos_y] = map(int, [pos_x, pos_y])
            img_x1[pos_x - 1][pos_y - 1] = float(x1)
            img_x2[pos_x - 1][pos_y - 1] = float(x2)
            img_y1[pos_x - 1][pos_y - 1] = float(y1)
            img_y2[pos_x - 1][pos_y - 1] = float(y2)

    # 每个纬度图片的每像素纬度差值不同，每行扫描标记

    print("开始标记：")
    # x是纵向大图位置
    for i in range(x_max):
        print("正在处理第 {} 行；".format(i+1))
        x_distance = abs(img_x2[i][0] - img_x1[i][0])
        y_distance = abs(img_y2[i][0] - img_y1[i][0])
        x_step = x_distance / 256  # x左右
        y_step = y_distance / 256  # y上下

        # y是横向大图位置
        for j in range(y_max):

            # 每个GPS数据对比
            for g in range(len(event_p[0])):

                # if start_x <= event_x[i] <= end_x and start_y >= event_y[i] >= end_y
                if img_x1[i][j] <= event_p[0][g] <= img_x2[i][j] and img_y1[i][j] >= event_p[1][g] >= img_y2[i][j]:
                    gps_x = int((event_p[0][g] - img_x1[i][j]) / x_step)
                    gps_y = int((img_y1[i][j] - event_p[1][g]) / y_step)
                    img_x_x = gps_y         # 经纬度和图像坐标计算，x和y正好是相反的
                    img_y_y = gps_x

                    # string = [["大图x"], ["大图y"], ["图中x"], ["图中y"]]
                    string[0].append(i)
                    string[1].append(j)
                    string[2].append(img_x_x)
                    string[3].append(img_y_y)

    print("处理完毕。保存csv文件。")
    # 将标签信息写入文件
    with open(map_gps_dir, "w", newline='') as file:
        c_writer = csv.writer(file)
        for i in range(len(string[0])):
            c_writer.writerow([string[0][i], string[1][i], string[2][i], string[3][i]])
    print("{} 保存完毕！".format(map_gps_dir))


def generate_image_per_pix():

    # 读入map_gps信息
    print("{} 开始分配内存...".format(datetime.datetime.now()))
    map_label = np.zeros((256 * x_max, 256 * y_max), dtype=np.uint8)

    print("{} 正在读取标签数据（共79316条）...".format(datetime.datetime.now()))
    with open(map_gps_dir, "r") as f:
        c_reader = csv.reader(f)
        csv_line_num = 0
        for item in c_reader:
            csv_line_num += 1
            if csv_line_num == 1:
                continue
            [gx1, gy1, gx2, gy2] = map(int, item)

            # 将每行数据映射成map上的次数
            xx = 256 * gx1 + gx2
            yy = 256 * gy1 + gy2
            map_label[xx][yy] += 1

    # 共产生 256*(174-1)+1  *  256*(222-1)+1 张图片
    step = 2
    times = int(256/step)
    print("{} 开始生成图片对应标签：".format(datetime.datetime.now()))
    f_dir, f_name = os.path.split(map_gps_dir)
    per_pix_label = os.path.join(f_dir, "{}_pix_label.csv".format(step))
    with open(per_pix_label, "w", newline='') as file:
        c_writer = csv.writer(file)
        c_writer.writerow(["name:", "label:"])

        for j in range(times * (x_max - 1) + 1):
            print("{} 正在处理第{}行图片({}%)".format(datetime.datetime.now(), j+1,
                                              100 * round(float(j)/(times * (x_max - 1) + 1), 4)))

            for i in range(times * (y_max - 1) + 1):

                # 计算标签
                map_name = "{}-{}.jpg".format(j + 1, i + 1)
                # map_label_temp = map_label[j:j + 256, i: i+256]
                map_label_temp = map_label[j*step:j*step+256, i*step: i*step+256]
                label = np.sum(map_label_temp)
                c_writer.writerow([map_name, label])

    print("{} 保存完毕！".format(per_pix_label))


if __name__ == "__main__":

    train_label_path = "D:/map_data/train/train.csv"
    NYPD_dir = "./NYPD_train.csv"

    time_now = datetime.datetime.now()

    # map_gps(train_label_path, NYPD_dir)
    generate_image_per_pix()

    time_end = datetime.datetime.now()
    print("耗时：" + str(time_end - time_now))
