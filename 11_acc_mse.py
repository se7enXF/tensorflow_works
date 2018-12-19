# -*- coding:utf-8 -*-  
import csv
import sys
import time

prediction_result = "/home/se7en/my_caffe/python/caffe_models/l_r4-8_result.csv"
test_labels ="/home/se7en/my_caffe/python/caffe_models/test_labes.csv"
result_number = 6800

csvFile = open(prediction_result,"r")
csv_reader = csv.reader(csvFile)
csv_line_num = 0
key = 0

zero = 0
one = 0
two = 0

for line in csv_reader:
	csv_line_num += 1
	#if csv_line_num == 1:
	#	continue
	print(line[0],line[1],"......",csv_line_num,)
	print("/",result_number,"|",str(round(100*float(csv_line_num)/result_number,2)),"%")
	
	csvfile = open(test_labels,'r')
	csvreader=csv.reader(csvfile)
	for row in csvreader:
		img_lab = row[1]
		img_name = row[0]
		
		if img_name == line[0]:
			key = 1
			dis = abs(int(line[1])-int(img_lab)) 
			if dis == 0:
				zero += 1
			elif dis == 1:
				one += 1
			elif dis == 2:
				two += 1
			else:
				sys.exit("Wrong label!")
			break
		else:
			key = 0
		
	csvfile.close
	if key == 0:
		sys.exit("Wrong filename!")		
csvFile.close
print ("误差0,1,2依次为:",zero,one,two)
print ("预测精确度:",str(round(100*float(zero)/result_number,2)),"%")
