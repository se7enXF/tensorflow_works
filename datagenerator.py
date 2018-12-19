import numpy as np
import cv2
import tensorflow as tf

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""


class ImageDataGenerator:
    def __init__(self, class_list, horizontal_flip=False, shuffle=False, 
                 n_class=3, scale_size=(227, 227)):

        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = n_class
        self.shuffle = shuffle
        self.mean = np.array([104., 117., 124.])
        self.scale_size = scale_size
        self.pointer = 0
        self.read_class_list(class_list)
        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                if self.n_classes == 0:
                    self.labels.append(float(items[1]))
                else:
                    self.labels.append(int(items[1]))

            # store total number of data
            self.data_size = len(self.labels)
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.images.copy()
        labels = self.labels.copy()
        self.images = []
        self.labels = []
        
        # create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]
        
        # update pointer
        self.pointer += batch_size
        
        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            img = cv2.imread(paths[i])
            
            # flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)
            
            # rescale image
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img = img.astype(np.float32)
            
            # subtract mean
            img -= self.mean
                                                                 
            images[i] = img

        # Expand labels to one hot encoding
        if self.n_classes == 0:
            one_hot_labels = np.zeros(batch_size)
            for i in range(len(labels)):
                one_hot_labels[i] = labels[i]
        else:
            one_hot_labels = np.zeros((batch_size, self.n_classes))
            for i in range(len(labels)):
                one_hot_labels[i][labels[i]] = 1

        # return array of images and labels
        return images, one_hot_labels


def my_image_list(flies):
    with open(flies) as f:
        lines = f.readlines()
        images = []
        labels = []
        for l in lines:
            items = l.split()
            images.append(items[0])
            labels.append(int(items[1]))

        # shuffle
        temp = np.array([images, labels])
        temp = temp.transpose()
        np.random.shuffle(temp)
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [int(float(i)) for i in label_list]

        return image_list, label_list


def get_batch(image_list, label_list, img_width, img_height, batch_size):

    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              num_threads=64, capacity=batch_size*3)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels
