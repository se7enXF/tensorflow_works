# -*- coding:utf-8 -*-  
import csv

prediction_result = "reg_0.0001_predict.csv"
test_labels = "test_label_norma.csv"

t_n = []
t_l = []
p_n = []
p_l = []

print("读取原始结果......" + test_labels)
with open(test_labels, "r") as File:
	p_reader = csv.reader(File)
	csv_cp = list(p_reader)
	csv_line_num = 0
	for line in csv_cp:
		csv_line_num += 1
		if csv_line_num == 1:
			continue
		t_n.append(line[0])
		t_l.append(line[1])

print("读取预测结果......" + prediction_result)
with open(prediction_result, "r") as File:
	p_reader = csv.reader(File)
	csv_cp = list(p_reader)
	csv_line_num = 0
	for line in csv_cp:
		csv_line_num += 1
		if csv_line_num == 1:
			continue
		p_n.append(line[0])
		p_l.append(line[1])

print("计算MSE......" )
error = 0
for i in range(len(t_n)):
	t_name = t_n[i]
	t_label = float(t_l[i])

	for j in range(len(p_n)):
		p_name = p_n[j]
		p_label = float(p_l[j])
		if t_name == p_name:
			error += pow(p_label - t_label, 2)
			break

MSE = error/len(t_n)
print("预测结果 {} 的MSE是 {}".format(prediction_result, round(MSE, 4)))

