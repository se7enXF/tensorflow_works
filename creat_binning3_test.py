# -*- coding:utf-8 -*-
'''

'''
import os
import csv



tmp_counter = [0, 0, 0]


def img_classfication(csv_label, csv_bin3):
	result = [["图片名"], ["标签"]]
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
			lable = int(item[7])
			if lable <= 0:
				tmp_counter[0] += 1
				result[0].append(img_name)
				result[1].append(0)
			elif lable <= 2:
				tmp_counter[1] += 1
				result[0].append(img_name)
				result[1].append(1)
			else:
				tmp_counter[2] += 1
				result[0].append(img_name)
				result[1].append(2)
			if csv_line_num % 10 == 0:
				print("正在分类......" + str(round(csv_line_num / line_sum, 4) * 100) + "%")

		with open(csv_bin3, "w", newline="") as F:
			csv_writer = csv.writer(F)
			for i in range(len(result[0])):
				csv_writer.writerow([result[0][i], result[1][i]])
		print("预测结果保存在：" + csv_bin3)


if __name__ == "__main__":
	test_csv = "G:/tf_work/test_label.csv"
	dst_dir = "./test.csv"

	img_classfication(test_csv, dst_dir)

