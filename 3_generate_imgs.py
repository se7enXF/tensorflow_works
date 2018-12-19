# -*- coding:utf-8 -*-
'''
用途：产生插值的图片。
命令行参数：
输出：data/生成对应.csv文件，data/train_all/生成插值图片
'''

import numpy as np
import csv
from PIL import Image
import matplotlib.pylab as plt
import datetime
import os
import random
from multiprocessing.dummy import Pool as ThreadPool
import gc

x_max = 174     # 1~174（高）
y_max = 222     # 1~222（宽）


img_name_buffer = [[0 for i in range(222)] for i in range(174)]
h_label = [["图片名"], ["标签"]]


def generate_image():
    select_imgs = "D:/data/2_selected_labels.csv"
    dis_dir = "D:/data/2_selected_train"

    img_dir = "D:/map_data/train/train"
    label_path = "D:/map_data/train/train.csv"
    if not os.path.exists(dis_dir):
        os.mkdir(dis_dir)

    print("{} Reading all images data...".format(datetime.datetime.now()))
    with open(label_path, "r") as File:
        c_reader = csv.reader(File)
        csv_line_num = 0
        for item in c_reader:
            csv_line_num += 1
            if csv_line_num == 1:
                continue

            # 读入图片信息
            img_name = item[0]
            pos_x = item[5]
            pos_y = item[6]
            [pos_x, pos_y] = map(int, [pos_x, pos_y])
            img_name_buffer[pos_x - 1][pos_y - 1] = img_name

    # 将整个卫星图读入内存
    print("{} Load all map into memory:".format(datetime.datetime.now()))

    # full_dis = np.zeros([174*256, 222*256, 3], dtype=np.uint8)
    # 先将第一行读入，后续图像方便后续连接
    print("{} 读取第 1 行图像...".format(datetime.datetime.now()))
    full_dis = plt.imread(os.path.join(img_dir, str(img_name_buffer[0][0])))        # 第1行，第1张
    for y in range(1, y_max):
        img1 = plt.imread(os.path.join(img_dir, str(img_name_buffer[0][y])))        # 第x行，第y张
        full_dis = np.concatenate((full_dis, img1), axis=1)                         # 横向拼接一列

    for x in range(1, x_max):                                                       # 每一行
        print("{} 读取第 {} 行图像...".format(datetime.datetime.now(), x+1))
        dis1 = plt.imread(os.path.join(img_dir, str(img_name_buffer[x][0])))        # 第x行，第1张
        for y in range(1, y_max):
            img1 = plt.imread(os.path.join(img_dir, str(img_name_buffer[x][y])))    # 第x行，第y张
            dis1 = np.concatenate((dis1, img1), axis=1)                             # 横向拼接一列

        # 然后和之前一行纵向拼接
        full_dis = np.concatenate((full_dis, dis1), axis=0)

    # 图像已经读到内存，然后进行裁剪
    print("{} All images have loaded.Reading selected images data:".format(datetime.datetime.now()))

    sel_data = [[], [], []]  # 图片起始左上角x，y，label
    with open(select_imgs, "r") as File:
        c_reader = csv.reader(File)
        csv_line_num = 0
        for item in c_reader:
            csv_line_num += 1
            if csv_line_num == 1:
                continue

            # 读入图片信息
            [img_name, img_label] = item
            f_name, f_type = os.path.splitext(img_name)
            f_x, f_y = f_name.split("-")
            sel_data[0].append(f_x)
            sel_data[1].append(f_y)
            sel_data[2].append(img_label)

    print("{} Data loaded,start generating new images...".format(datetime.datetime.now()))

    step = 2
    times = int(256/step)
    for s in range(len(sel_data[0])):

        img_sx = int(sel_data[0][s])-1
        img_sy = int(sel_data[1][s])-1
        # 裁剪图像
        dis_img = full_dis[img_sx*step:img_sx*step+256, img_sy*step:img_sy*step+256, :]
        img2 = Image.fromarray(dis_img)

        img2_name = "{}-{}.jpg".format(sel_data[0][s], sel_data[1][s])
        img2.save(os.path.join(dis_dir, img2_name))

    print("{} OK!Images have saved to:{}".format(datetime.datetime.now(), dis_dir))


def select_images():
    pix_2_dir = "D:/data/2_pix_label.csv"

    write_str = [["name:"], ["label"]]
    image_data = []

    print("{} Reading csv and select images for class_0".format(datetime.datetime.now()))
    with open(pix_2_dir, "r") as File:
        c_reader = csv.reader(File)
        line_count = 0
        for item in c_reader:
            line_count += 1
            if line_count == 1:
                continue

            img_n, img_l = item
            if int(img_l) == 0:
                image_data.append(img_n)
    t_data = random.sample(image_data, 2000)
    del image_data      # release memory
    gc.collect()

    for cc in t_data:
        write_str[0].append(cc)
        write_str[1].append(0)

    # select left 1-99
    print("{} Reading csv for the left class".format(datetime.datetime.now()))
    data_buf = [[], []]
    with open(pix_2_dir, "r") as File:
        c_reader = csv.reader(File)
        line_count = 0
        for item in c_reader:
            line_count += 1
            if line_count == 1:
                continue

            img_n, img_l = item
            if int(img_l) == 0:
                continue
            data_buf[0].append(img_n)
            data_buf[1].append(img_l)

    for i in range(1, 138):
        print("{} Select images...class_{}".format(datetime.datetime.now(), i))

        image_data = []
        for dt in range(len(data_buf[0])):
            if int(data_buf[1][dt]) == i:
                image_data.append(data_buf[0][dt])

        if len(image_data) >= 2000:
            t_data = random.sample(image_data, 2000)
        else:
            t_data = image_data
        for cc in t_data:
            write_str[0].append(cc)
            write_str[1].append(i)

    print("{} Select images...class_138".format(datetime.datetime.now()))
    image_data = []
    for dt in range(len(data_buf[0])):
        if int(data_buf[1][dt]) == 185:
            image_data.append(data_buf[0][dt])

    if len(image_data) >= 2000:
        t_data = random.sample(image_data, 2000)
    else:
        t_data = image_data
    for cc in t_data:
        write_str[0].append(cc)
        write_str[1].append(138)

    csv_dir = "D:/data/2_selected_labels.csv"
    with open(csv_dir, "w", newline='') as file:
        c_writer = csv.writer(file)
        for i in range(len(write_str[0])):
            c_writer.writerow([write_str[0][i], write_str[1][i]])
    print("{} 保存完毕！".format(csv_dir))


def select_images_mult_thread():

    pix_2_dir = "D:/data/2_pix_label.csv"
    print("{} Reading image csv...".format(datetime.datetime.now()))
    write_str = [["name:"], ["label"]]

    def my_thread(s_):
        for i in range(s_, s_+25):
            print("{} Select images...class_{}".format(datetime.datetime.now(), i))

            image_data = []
            with open(pix_2_dir, "r") as File:
                c_reader = csv.reader(File)
                line_count = 0
                for item in c_reader:
                    line_count += 1
                    if line_count == 1:
                        continue

                    img_n, img_l = item
                    if int(img_l) == i:
                        image_data.append(img_n)
            t_data = random.sample(image_data, 100)
            for cc in t_data:
                write_str[0].append(cc)
                write_str[1].append(i)

    items = [0, 25, 50, 75]
    pool = ThreadPool()
    pool.map(my_thread, items)
    pool.close()
    pool.join()

    print("{} All threads finish!Saving results...".format(datetime.datetime.now()))
    csv_dir = "D:/data/2_selected_label.csv"
    with open(csv_dir, "w", newline='') as file:
        c_writer = csv.writer(file)
        for i in range(len(write_str[0])):
            c_writer.writerow([write_str[0][i], write_str[1][i]])
    print("{} 保存完毕！".format(csv_dir))


if __name__ == "__main__":

    time_now = datetime.datetime.now()

    # select_images()
    generate_image()

    time_end = datetime.datetime.now()
    print("耗时：" + str(time_end - time_now))
