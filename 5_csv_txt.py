# -*- coding:utf-8 -*-
'''
用途：将发生次数按照比例分为n类
命令行参数：无
输出：将对应类别拷贝到对应文件夹（train/data/不同类别文件夹）
'''
import os
import datetime
import csv
import shutil
import math

class_n = 3
tmp_counter = [0, 0, 0]
tmp_txt = [[], []]

tmp_train = [[], []]
tmp_val = [[], []]


def img_classfication(csv_label):

	dir_, file_ = os.path.split(csv_label)
	# train_bin3 = os.path.join(dir_, "train_bin3.csv")
	train_bin3 = "./class_test_bin3.csv"
	record_buffer = [["文件名"], ["类别"]]

	with open(csv_label, "r") as File:
		csv_reader = csv.reader(File)
		csv_cp = list(csv_reader)
		line_sum = len(csv_cp)
		csv_line_num = 0
		print("读取图片信息......"+csv_label)
		for item in csv_cp:
			csv_line_num += 1
			if csv_line_num == 1:
				continue
			# 开始读取数据，每一行结构[图片名，标签]
			img_name = item[0]
			# train:1,test:7
			lable = float(item[7])

			# train:110,test:
			if lable <= 0:
				tmp_counter[0] += 1
				record_buffer[0].append(img_name)
				record_buffer[1].append("0")
			# train:250,test:
			elif lable <= 1:
				tmp_counter[1] += 1
				record_buffer[0].append(img_name)
				record_buffer[1].append("1")
			else:
				tmp_counter[2] += 1
				record_buffer[0].append(img_name)
				record_buffer[1].append("2")
			if csv_line_num % 100 == 0:
				print("正在分类......" + str(round(csv_line_num / line_sum, 4) * 100) + "%")

	print(tmp_counter, "保存分类csv到：", train_bin3)
	with open(train_bin3, "w", newline="") as csv_file:
		csv_writer = csv.writer(csv_file)
		for i in range(len(record_buffer[0])):
			csv_writer.writerow([record_buffer[0][i], record_buffer[1][i]])


def train_label_normalization():

	train_csv = "D:/data/2_selected_labels.csv"
	train_path = "D:/data/2_selected_train"
	path, f_name = os.path.split(train_csv)
	norma_label = os.path.join(path, "norma_label.txt")

	with open(train_csv, "r") as File:
		csv_reader = csv.reader(File)
		for i, item in enumerate(csv_reader):
			if i == 0:
				continue

			# 开始读取数据，每一行结构[图片名，标签]
			img_name = os.path.join(train_path, item[0])
			lable = float(item[1])
			tmp_txt[0].append(img_name)
			tmp_txt[1].append(lable/138)

	with open(norma_label, "w") as txt_f:
		for i in range(len(tmp_txt[0])):
			string = tmp_txt[0][i]+" "+str(tmp_txt[1][i])+"\n"
			txt_f.write(string)


def test_label_normalization():

	csv_label = "D:\\map_data\\test\\test_label.csv"
	path, f_name = os.path.split(csv_label)
	norma_label = os.path.join(path, "test_label_norma.csv")

	with open(csv_label, "r") as File:
		csv_reader = csv.reader(File)
		for i, item in enumerate(csv_reader):
			if i == 0:
				continue

			# 开始读取数据，每一行结构[图片名，标签]
			# img_name = os.path.join("D:/map_data/test/test", item[0])
			img_name = item[0]
			lable = float(item[7])
			tmp_txt[0].append(img_name)
			tmp_txt[1].append(round(lable/100, 4))

	with open(norma_label, "w", newline="") as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["图片名", "标签"])
		for i in range(len(tmp_txt[0])):
			csv_writer.writerow([tmp_txt[0][i], tmp_txt[1][i]])
	print("Saved to:"+norma_label)


def csv_2_txt(csv_label):
	path, f_name = os.path.split(csv_label)
	train_txt = os.path.join(path, "train.txt")
	val_txt = os.path.join(path, "val.txt")
	counter = [0 for i in range(186)]		# 最大标签1255

	with open(csv_label, "r") as File:
		csv_reader = csv.reader(File)
		for i, item in enumerate(csv_reader):
			if i == 0:
				continue

			# 开始读取数据，每一行结构[图片名，标签]
			img_name = os.path.join("G:/data/train_all", item[0])
			lable = int(item[1])
			# tmp_txt[0].append(img_name)
			# if counter[lable] <= 10:
			# 	tmp_val[0].append(img_name)
			# 	tmp_val[1].append(lable)
			# 	counter[lable] += 1
			# 	continue

			tmp_train[0].append(img_name)
			tmp_train[1].append(lable)

	with open(train_txt, "w") as txt_f:
		for i in range(len(tmp_train[0])):
			string = tmp_train[0][i]+" "+str(tmp_train[1][i])+"\n"
			txt_f.write(string)
	# with open(val_txt, "w") as txt_f:
	# 	for i in range(len(tmp_val[0])):
	# 		string = tmp_val[0][i]+" "+str(tmp_val[1][i])+"\n"
	# 		txt_f.write(string)

	print("Saved to:", train_txt, val_txt)


if __name__ == "__main__":

	start_time = datetime.datetime.now()

	# train_label_normalization()
	test_label_normalization()

	end_time = datetime.datetime.now()
	print("耗时：" + str(end_time - start_time))
