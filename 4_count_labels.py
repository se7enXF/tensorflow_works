# -*- coding:utf-8 -*-  
import os
import csv

train_label = "D:/data/2_selected_labels.csv"

max_count = 0
label_array = [0 for x in range(0, 186)]		# 186-1train,235-1test

with open(train_label, "r") as F:
	print("打开文件:{}".format(train_label))
	c_reader = csv.reader(F)

	csv_line_num = 0
	line_sum = 156627585
	for item in c_reader:
		csv_line_num += 1
		if csv_line_num == 1:
			continue

		img_lab = int(item[1])
		if int(img_lab) > max_count:
			max_count = int(img_lab)
		label_array[int(img_lab)] += 1
		if csv_line_num % 10000 == 0:
			print("正在处理第{}个数据...({}%)".format(csv_line_num, round(csv_line_num/line_sum, 4)*100))

print("max_count=", max_count)
print(label_array)
print("地区总数:{}".format(sum(label_array)))

file_dir, file_name = os.path.split(train_label)
wave_csv = file_name[:-9]+"wave.csv"
wave_csv = os.path.join(file_dir, wave_csv)
# print(file_name)
# exit()
with open(wave_csv, "w", newline="") as F:
	csv_writer = csv.writer(F)
	csv_writer.writerow(["事故次数", "图片数量"])
	for i in range(len(label_array)):
		csv_writer.writerow([i, label_array[i]])

# 将事故发生次数和地区数统计出来
