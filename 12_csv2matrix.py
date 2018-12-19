# -*- coding:utf-8 -*-  
import csv
import os


def label_2_matrix(csv_dir, matrix_dir):
    string = [[0 for i in range(88)] for i in range(115)]
    with open(csv_dir, "r") as File:
        p_reader = csv.reader(File)
        csv_cp = list(p_reader)
        line_sum = len(csv_cp)
        csv_line_num = 0
        print("读取预测结果......" + csv_dir)
        for line in csv_cp:
            csv_line_num += 1
            if csv_line_num == 1:
                continue

            img_name = line[0]
            label = int(line[1])
            pos = img_name[6:]
            pos, ext = os.path.splitext(pos)
            x, y = pos.split("-")

            string[int(x)-1][int(y)-1] = label

        with open(matrix_dir, "w", newline="") as C:
            wt = csv.writer(C)
            for i in range(114):
                wt.writerow(string[i])
        print("Matrix 保存完毕："+matrix_dir)


if __name__ == "__main__":
    prediction_result = "./h1_predict.csv"
    test_labels = "./test.csv"

    p_matrix = "./result/h1_p_matrix.csv"
    t_matrix = "./result/t_matrix.csv"

    label_2_matrix(prediction_result, p_matrix)
    # label_2_matrix(test_labels, t_matrix)
