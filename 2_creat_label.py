# -*- coding:utf-8 -*-
'''
用途：对数据集进行标注，【文件名，左上角x，左上角y，右下角x，右下角y，图片位置x，图片位置y，事故发生次数】
    写入对应的.csv文件，以备后续处理。
输出：当前目录生成对应.csv文件
'''
import csv
import os
import datetime
import sys

img_event = ["事故发生次数"]


def add_label(NYPD, img_csv, map_kind):

    total = 0
    f_dir, f_name = os.path.split(img_csv)
    labeled_dir = os.path.join(f_dir, map_kind + "_label.csv")

    # 读入图片坐标信息
    if not os.path.exists(img_csv):
        print("路径错误！")
        exit()
    csvFile = open(img_csv, "r")
    csv_reader = csv.reader(csvFile)
    csv_cp = list(csv_reader)
    item_sum = len(csv_cp)

    # 读入NPYPD文件
    event_x, event_y = [], []
    File = open(NYPD, "r")
    c_reader = csv.reader(File)
    csv_line_num = 0
    for item in c_reader:
        csv_line_num += 1
        if csv_line_num == 1:
            continue

        event_x.append(float(item[3]))
        event_y.append(float(item[2]))

    print("开始标记数据......")
    line_number = 0
    for line_item in csv_cp:
        # print(line_item)
        line_number += 1
        # 跳过首行
        if line_number == 1:
            continue

        # 每一个图片信息中：
        label = 0
        [img_name, start_x, start_y, end_x, end_y, X, Y] = line_item
        [start_x, start_y, end_x, end_y] = map(float, [start_x, start_y, end_x, end_y])

        # 逐行对比
        for i in range(len(event_x)):
            if start_x <= event_x[i] <= end_x and start_y >= event_y[i] >= end_y:
                label += 1
                total += 1
        img_event.append(label)
        if line_number % 100 == 0:
            print("正在标记数据......" + str(round(line_number / item_sum, 4) * 100) + "%")

    print("地图内事故次数总和：{}".format(total))
    # 将原始数据和标记结果合并保存
    print("正在写入文件{}".format(labeled_dir))
    with open(labeled_dir, "w", newline='') as new_csv:
        csv_writer = csv.writer(new_csv)
        for i in range(len(csv_cp)):
            csv_cp[i].append(img_event[i])
            csv_writer.writerow(csv_cp[i])


if __name__ == '__main__':
    NYPD_train = "NYPD_train.csv"
    NPPD_test = "NYPD_all.csv"
    csv_train = "G:/map_data/train/train.csv"
    csv_test = "G:/test/test.csv"
    if len(sys.argv) < 2:
        print("缺少参数：train 或 test")
        exit()
    tp = sys.argv[1]
    if tp == "train":
        NYPD_path = NYPD_train
        csv_path = csv_train
        map_kind = "train"
    elif tp == "test":
        NYPD_path = NPPD_test
        csv_path = csv_test
        map_kind = "test"
    else:
        print("参数错误！train 或 test")
        exit()

    start_time = datetime.datetime.now()
    add_label(NYPD_path, csv_path, map_kind)
    end_time = datetime.datetime.now()
    print("完成！所用时间：" + str(end_time - start_time))

