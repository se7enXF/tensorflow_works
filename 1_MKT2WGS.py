# -*- coding:utf-8 -*-
'''
用途：train 或 test 图片集数据处理，【文件名，左上角x，左上角y，右下角x，右下角y，图片位置x，图片位置y】
    写入对应的.csv文件，以备后续处理。
命令行参数：train 或 test
输出：图片文件夹当前目录生成对应.csv文件
'''
import os
import math
import datetime
import sys
import csv

mercator = [0, 0]           # 摩卡托坐标
lonLat = [0, 0]             # 经纬度坐标

r_name = ["文件名"]
r_start_x = ["左上角x"]
r_start_y = ["左上角y"]
r_end_x = ["右下角x"]
r_end_y = ["右下角y"]
img_x = ["图片位置x"]
img_y = ["图片位置y"]


def web_mercator2lonlat(mct):   # Web墨卡托转经纬度
    px = mct[0]/20037508.34*180
    py = mct[1]/20037508.34*180
    py = 180/math.pi*(2*math.atan(math.exp(py*math.pi/180))-math.pi/2)
    lonLat[0] = px
    lonLat[1] = py
    return lonLat


def transform(img_path, map_kind):
    files = os.listdir(img_path)
    sum_img = len(files)/2
    tmp = sum_img / 100
    img_count = 0
    print("开始转换......" + "（总共图片数量：" + str(int(sum_img)) + ")")

    if map_kind == "train":
        start_line = 12
        end_line = 14
    elif map_kind == "test":
        start_line = 11
        end_line = 13

    for file in files:
        name, tail_name = os.path.splitext(file)
        if tail_name == ".txt":
            line_num = 0
            img_count += 1
            r_name.append(name + ".tif")
            name_list = name.split("_")
            position = name_list[1].split("-")
            img_x.append(position[0])
            img_y.append(position[1])
            # 打开.txt坐标文件
            with open(img_path+"/"+file, 'r') as fd:
                for line in fd:
                    line_num += 1

                    # 左上角坐标
                    if line_num == start_line:
                        mercator[0] = float(line[6:30])
                        mercator[1] = float(line[31:])
                        lonLat1 = web_mercator2lonlat(mercator)
                        r_start_x.append(lonLat1[0])
                        r_start_y.append(lonLat1[1])
                        # print 'new 左上角经纬:',x1,',',y1

                    # 右下角坐标
                    if line_num == end_line:
                        x = float(line[6:30])
                        y = float(line[31:])
                        mercator[0] = x
                        mercator[1] = y
                        lonLat2 = web_mercator2lonlat(mercator)
                        r_end_x.append(lonLat2[0])
                        r_end_y.append(lonLat2[1])
                        # print 'new 右下角经纬:',x2,',',y2

            if tmp <= img_count:
                tmp += sum_img / 100
                print("正在转换坐标......" + str(round(img_count/sum_img, 4)*100) + "%")

    print("正在写入csv文件......" + str(img_path+"/../" + map_kind + ".csv"))
    new_csv = open(img_path+"/../" + map_kind + ".csv", "w", newline='')
    csv_writer = csv.writer(new_csv)
    string = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(r_name)):
        string[0] = r_name[i]
        string[1] = r_start_x[i]
        string[2] = r_start_y[i]
        string[3] = r_end_x[i]
        string[4] = r_end_y[i]
        string[5] = img_x[i]
        string[6] = img_y[i]
        csv_writer.writerow(string)


if __name__ == '__main__':
    train_path = "G:/train/train"
    test_path = "G:/test/test"
    if len(sys.argv) < 2:
        print("缺少参数：train 或 test")
        exit()
    tp = sys.argv[1]
    if tp == "train":
        img_path = train_path
        map_kind = "train"
    elif tp == "test":
        img_path = test_path
        map_kind = "test"
    else:
        print("参数错误！train 或 test")
        exit()

    start_time = datetime.datetime.now()
    transform(img_path, map_kind)
    end_time = datetime.datetime.now()

    print("完成！所用时间：" + str(end_time - start_time))
