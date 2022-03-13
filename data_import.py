import os
import numpy as np
import tensorflow as tf

def getBase(drive_num):
    return f'./data/2011_09_26/2011_09_26_drive_000{drive_num}_sync/'

def getImages(folders):
    data_dir = 'image_00/data/'
    X = []
    for folder in folders:
        base = getBase(folder)
        files = os.listdir(base + data_dir)
        files.sort()
        # print(len(files))
        for file in files:
            with open(base + data_dir + file, 'rb') as infile:
                X.append([infile.read()])
    return np.array(X)

def getLabels(folders):
    data_dir = 'oxts/data/'
    Y = []
    for folder in folders:
        base = getBase(folder)
        files = os.listdir(base + data_dir)
        files.sort()
        # print(len(files))
        for file in files:
            with open(base + data_dir + file, 'r') as infile:
                labels = infile.readline().strip().split(' ')
                Y.append([float(labels[5]), float(labels[14])])
    return np.array(Y)

folders = [1, 2, 5]

X = getImages(folders)
Y = getLabels(folders)
print(X.shape, Y.shape)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def example(image, response):
    feature = {
        'image': _bytes_feature(image),
        'yaw': _float_feature(response[0]),
        'forward_accel': _float_feature(response[1])
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

record_file = './data/real/images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for image, response in zip(X, Y):
        tf_example = example(image[0], response)
        writer.write(tf_example.SerializeToString())