# -*- coding:utf-8 -*-
'''
用途：将每一类图片数目统一
命令行参数：无
输出：无
'''
import os
import datetime


def delete_file(folder, leave):
	imgs = os.listdir(folder)
	sum_imgs = len(imgs)
	del_n = sum_imgs - leave
	print("开始统一数据：" + folder)
	# 等差删除
	line_counter = 0
	delete_counter = 0
	step = int(sum_imgs/del_n)
	# print(step)
	for i in imgs:
		line_counter += 1
		if delete_counter < del_n and line_counter % step == 0:
			delete_counter += 1
			del_path = os.path.join(folder, i)
			os.remove(del_path)

	print("Delete:"+str(delete_counter)+"......Leave:"+str(leave))


if __name__ == "__main__":
	leave_n = 39000
	data_dir = "G:/train/data"
	class0_folder = os.path.join(data_dir, "class_"+str(0))
	class1_folder = os.path.join(data_dir, "class_" + str(1))
	class2_folder = os.path.join(data_dir, "class_" + str(2))

	start_time = datetime.datetime.now()
	delete_file(class0_folder, leave_n)
	delete_file(class1_folder, leave_n)
	delete_file(class2_folder, leave_n)

	end_time = datetime.datetime.now()
	print("完成！所用时间:"+str(end_time-start_time))
