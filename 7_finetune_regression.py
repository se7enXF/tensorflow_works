# encoding:utf-8 #
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator

"""
Configuration settings
"""

# Path to the text files for the trainings and validation set
train_file = "D:/data/norma_label.txt"

# Learning params
learning_rate = 0.001
num_epochs = 6
batch_size = 256

# Network params
dropout_rate = 0.5
train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
summary_step = 5

# Path for tf.summary.FileWriter and to store model checkpoints
restore_dir = "D:/tf_work/log/reg_2019-01-04_14-34_lr_0.001_restore"
if restore_dir:
    filewriter_path = "D:/tf_work/log/reg_{}_lr_{}_restore".format(datetime.now().strftime("%Y-%m-%d_%H-%M"), learning_rate)
else:
    filewriter_path = "D:/tf_work/log/reg_{}_lr_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M"), learning_rate)

    # Create parent path if it doesn't exist
    if not os.path.isdir(filewriter_path):
        os.mkdir(filewriter_path)

checkpoint_path = filewriter_path


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, train_layers, n_class=0)

# Link variable to model output
score = model.fc8
score = tf.reshape(score, [batch_size])

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("loss"):
    loss = tf.losses.mean_squared_error(score, y)

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
tf.summary.scalar('MSE', loss)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, horizontal_flip=True, shuffle=True, n_class=0)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)

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
    if restore_dir:
        print("{} Restore from {}".format(datetime.now(), restore_dir))
        saver.restore(sess, tf.train.latest_checkpoint(restore_dir))
    else:
        weight_path = "hybird.npy"
        model.load_my_weights(weight_path, sess)

    print("{} Start training...".format(datetime.now()))
    print("{} tensorboard --logdir {}".format(datetime.now(), filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        step = 1
        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % summary_step == 0:
                mse = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("{} Step : {}    MSE: {}".format(datetime.now(), step, mse))
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
            step += 1

        # Reset the file pointer of the image data generator
        train_generator.reset_pointer()

        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
