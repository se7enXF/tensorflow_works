import os
import cv2
import numpy as np
import tensorflow as tf
import csv
from alexnet import AlexNet
import datetime

# edit here
test_img_path = "D:\\map_data\\test\\test"
ckpt_dir = "D:\\tf_work\\log\\reg_2018-12-20_15-18_lr_0.0001\\model_epoch3.ckpt"
num_class = 0

img_files = [os.path.join(test_img_path, f) for f in os.listdir(test_img_path) if f.endswith('.tif')]

print("{} loading images...".format(datetime.datetime.now()))
# load all images
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
imgs = []
for f in img_files:
    imgs.append(cv2.imread(f))

# placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

# create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, [], n_class=num_class)
saver = tf.train.Saver()

# define activation of last layer as score
score = model.fc8

# create op to calculate softmax
if num_class != 0:
    score = tf.nn.softmax(score)

print("{} Start predicting...".format(datetime.datetime.now()))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the weights into the model
    saver.restore(sess, ckpt_dir)

    results = [["图片名"], ["预测结果"]]
    # Loop over all images
    for i, image in enumerate(imgs):
        # Convert image to float32 and resize to (227x227)
        img = cv2.resize(image.astype(np.float32), (227, 227))

        # Subtract the ImageNet mean
        img -= imagenet_mean

        # Reshape as needed to feed into model
        img = img.reshape((1, 227, 227, 3))

        # Run the session and calculate the class probability
        probs = sess.run(score, feed_dict={x: img, keep_prob: 1})

        # Get the class name of the class with the highest probability

        if num_class == 0:
            img_path, img_name = os.path.split(img_files[i])
            results[0].append(img_name)
            p_label = round(probs[0][0], 4)
            if p_label < 0.0001:
                results[1].append(0)
            results[1].append(p_label)
            print(results[0][i], results[1][i])
        else:
            class_name = np.argmax(probs)
            img_path, img_name = os.path.split(img_files[i])
            results[0].append(img_name)
            results[1].append(class_name)

    result_dir = "./reg_0.0001_predict.csv"
    with open(result_dir, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in range(len(results[0])):
            csv_writer.writerow([results[0][i], results[1][i]])
    print("{} 预测结果保存在：{}".format(datetime.datetime.now(), result_dir))
