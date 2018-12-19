# -*- coding:utf-8 -*-
'''
用途：将分类好的图片分为训练集和验证集
命令行参数：
输出：在data下新建train,val文件夹，并将data目录中不同类别图片进行比例分离
'''
import os
import csv
import datetime


# 分离图片，添加路径
def divide_files(src_dir, img_fd):

    path, f_name = os.path.split(src_dir)
    train_path = os.path.join(path, "train.txt")
    val_path = os.path.join(path, "val.txt")
    test_path = os.path.join(path, "test.txt")
    train_buffer = [[], []]
    val_buffer = [[], []]
    test_buffer = [[], []]

    print("{} 分离为train,val,test".format(datetime.datetime.now()))
    for c in range(100):
        temp = []
        with open(src_dir, "r") as File:
            csv_reader = csv.reader(File)
            for i, item in enumerate(csv_reader):
                if i == 0:
                    continue

                f_name, f_label = item[0], int(item[1])
                f_name = os.path.join(img_fd, f_name)
                if f_label == c:
                    temp.append(f_name)

        for x in temp[0:60]:
            train_buffer[0].append(x)
            train_buffer[1].append(c)
        for x in temp[60:80]:
            val_buffer[0].append(x)
            val_buffer[1].append(c)
        for x in temp[80:100]:
            test_buffer[0].append(x)
            test_buffer[1].append(c)

    with open(train_path, "w") as ft:
        for i in range(len(train_buffer[0])):
            string = train_buffer[0][i] + " " + str(train_buffer[1][i]) + "\n"
            ft.write(string)
    with open(val_path, "w") as ft:
        for i in range(len(val_buffer[0])):
            string = val_buffer[0][i] + " " + str(val_buffer[1][i]) + "\n"
            ft.write(string)
    with open(test_path, "w") as ft:
        for i in range(len(test_buffer[0])):
            string = test_buffer[0][i] + " " + str(test_buffer[1][i]) + "\n"
            ft.write(string)


if __name__ == "__main__":
    data_dir = "D:/data/2_selected_label.csv"
    img_folder = "D:/data/2_selected_train"

    start_time = datetime.datetime.now()

    divide_files(data_dir, img_folder)

    end_time = datetime.datetime.now()
    print("完成!耗时："+str(end_time-start_time))
