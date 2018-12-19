import matplotlib.pyplot as plt
import csv
import os
from PIL import Image


def label_2_matrix(csv_dir, save_dir):
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    with open(csv_dir, "r") as File:
        p_reader = csv.reader(File)
        csv_cp = list(p_reader)
        line_sum = len(csv_cp)
        csv_line_num = 0
        print("读取数据......" + csv_dir)
        for line in csv_cp:
            csv_line_num += 1
            if csv_line_num == 1:
                continue

            img_name = line[0]
            label = int(line[1])
            pos = img_name[6:]
            pos, ext = os.path.splitext(pos)
            x, y = pos.split("-")
            if label == 0:
                x0.append(x)
                y0.append(y)
            elif label == 1:
                x1.append(x)
                y1.append(y)
            elif label == 2:
                x2.append(x)
                y2.append(y)
    size = 4
    mk = ","

    plt.xticks([])
    plt.yticks([])
    plt.scatter(x0, y0, color="g", label='0', marker=mk, s=size)
    plt.savefig("./result/0.png")

    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.scatter(x1, y1, color="b", label='1', marker=mk, s=size)
    plt.savefig("./result/1.png")

    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.scatter(x2, y2, color="r", label='2', marker=mk, s=size)
    plt.savefig("./result/2.png")

    plt.clf()
    # plt.savefig(save_dir)
    print("图片保存："+save_dir)



if __name__ == "__main__":
    prediction_result = "./predict.csv"
    test_labels = "./test.csv"
    p_save_path = "./result/predict.png"
    t_save_path = "./result/test.png"

    # label_2_matrix(prediction_result, p_save_path)
    label_2_matrix(test_labels, t_save_path)

    os.system("start explorer G:\\finetune_alexnet_with_tensorflow\\result")
