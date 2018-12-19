# -*- coding:utf-8 -*-  
import csv
import sys
import time

prediction_result = "./h_0.0001_predict.csv"
test_labels = "./test.csv"

hx_mareix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

with open(prediction_result, "r") as File:
	p_reader = csv.reader(File)
	csv_cp = list(p_reader)
	line_sum = len(csv_cp)
	csv_line_num = 0
	print("读取预测结果......" + prediction_result)
	for line in csv_cp:
		csv_line_num += 1
		if csv_line_num == 1:
			continue

		if csv_line_num % 10 == 0:
			print(line_sum, "|", str(round(100*float(csv_line_num)/line_sum, 2)), "%")

		with open(test_labels, "r") as F:
			p_reader = csv.reader(F)
			csv_cp = list(p_reader)
			line_sum = len(csv_cp)
			csv_line = 0
			for item in csv_cp:
				csv_line += 1
				if csv_line == 1:
					continue

				img_lab = int(item[1])
				img_name = item[0]
				if img_name == line[0]:
					hx_mareix[img_lab][int(line[1])] += 1

acc = int(hx_mareix[0][0]) + int(hx_mareix[1][1]) + int(hx_mareix[2][2])
total = acc+int(hx_mareix[0][1])+int(hx_mareix[0][2])+int(hx_mareix[1][0])\
		+int(hx_mareix[1][2])+int(hx_mareix[2][0])+int(hx_mareix[2][1])

print("混淆矩阵:")
print(hx_mareix)
print("预测精确度:"+str(acc)+"/"+str(total)+":"+str(round(100*float(acc)/total, 2)), "%")
