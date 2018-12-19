import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import *

"""
Configuration settings
"""
# Path for train data set
train_txt = "D:/data/train.txt"
val_txt = "D:/data/val.txt"

# Path to the text files for the trainings and validation set
train_file = train_txt
val_file = val_txt

# Learning params
learning_rate = 0.001
num_epochs = 5
batch_size = 50

# Network params
dropout_rate = 0.5
num_classes = 100
train_layers = ['fc8', 'fc7']

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "D:/tf_work/log/class_{}_lr_{}".format(datetime.now().strftime("%Y-%m-%d_%H_%M"), learning_rate)
checkpoint_path = filewriter_path

# How often we want to write the tf.summary data to disk
summary_step = 10

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.int32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
    tf.summary.histogram(var.name, var)
  
# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
t_image_list, t_label_list = my_image_list(train_txt)
train_image_batch, train_label_batch = get_batch(t_image_list, t_label_list, 227, 227, batch_size)
v_image_list, v_label_list = my_image_list(val_txt)
val_image_batch, val_label_batch = get_batch(v_image_list, v_label_list, 227, 227, batch_size)
train_batches_per_epoch = np.floor(6000/batch_size)
val_batches_per_epoch = np.floor(2000/batch_size)

# Start Tensorflow session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
 
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    # model.load_initial_weights(sess)
    weight_path = "hybird.npy"
    model.load_my_weights(weight_path, sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        step = 1
        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_ys = sess.run([train_image_batch, train_label_batch])
            batch_ys = onehot(batch_ys)

            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file

            if step % summary_step == 0:
                loss_record = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("{} Step : {}  LOSS:{}".format(datetime.now(), step, loss_record))
                s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(int(val_batches_per_epoch)):
            batch_xs, batch_ys = sess.run([val_image_batch, val_label_batch])
            batch_ys = onehot(batch_ys)
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
