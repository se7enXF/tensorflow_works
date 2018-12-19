# coding:utf-8
'''
用途：数据集整理工具，将图片转换为TFRecord保存和读取，显示
命令行参数：无（图片转换为TFRecord保存需要在主程序中修改）
输出：在train,val文件夹中生成对应.tfrecord文件
'''
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

BATCH_SIZE = 5
IMG_W = 227
IMG_H = 227


def img_2_tfrecords(input_img):
    # input_img="G:\train\data\train"
    tp = input_img.split(os.path.sep)[-1]
    output_tfrecord = os.path.join(input_img, tp+".tfrecords")
    classes = {'class_0', 'class_1', 'class_2'}
    writer = tf.python_io.TFRecordWriter(output_tfrecord)

    for index, name in enumerate(classes):
        class_path = os.path.join(input_img, name)
        item_counter = 0
        for img_name in os.listdir(class_path):
            item_counter += 1
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            img = img.resize((IMG_W, IMG_H))
            img_raw = img.tobytes()                                     # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))                                                         # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())                   # 序列化为字符串
            if item_counter % 500 == 0:
                print("正在将"+class_path+"--->转换为TFRecord......")
    writer.close()
    print(input_img+"转换完毕，TFRecord保存在："+output_tfrecord)


def read_tfrecord(tfrecord_dir):
    filename_queue = tf.train.string_input_producer([tfrecord_dir], shuffle=True)   # 读入tfrecord
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)                 # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })                               # 取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(image, [IMG_W, IMG_H, 3])                          # reshape为256*256的3通道图片
    img = tf.cast(img, tf.float32)                                      # 在流中抛出img张量
    lab = tf.cast(features['label'], tf.int32)                          # 在流中抛出label张量

    return img, lab


if __name__ == '__main__':

    data_dir = "G:/train/data"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    t_tfrecord_dir = "G:/train/data/train/train.tfrecords"
    v_tfrecord_dir = "G:/train/data/val/val.tfrecords"

    # img_2_tfrecords(train_dir)
    # exit()
    image_tensor, label_tensor = read_tfrecord(t_tfrecord_dir)      # 得到的数据是打乱的，但图片和标签一一对应

    img_batch, label_batch = tf.train.batch([image_tensor, label_tensor],
                                            batch_size=BATCH_SIZE,
                                            capacity=64)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img, label = sess.run([img_batch, label_batch])
        for i in range(5):
            print(label[i])
            img[i] = img[i]
            plt.imshow(img[i])
            plt.show()

        coord.request_stop()
        coord.join(threads)


