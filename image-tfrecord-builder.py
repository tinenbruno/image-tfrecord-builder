import numpy as np
import tensorflow as tf
import sys
import cv2

IMAGES_INPUT_FOLDER = "/home/bruno/Downloads/AVA_dataset/nature2/negative/"
DATASET_DEFINITIONS = r'../../AVA_dataset/AVA.txt'
OUTPUT_FILENAME = 'nature.tfrecords'

def load_image(path):
    image = cv2.imread(path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32)
    return None

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

file = open(DATASET_DEFINITIONS)
writer = tf.python_io.TFRecordWriter(OUTPUT_FILENAME)

for line in file:
    line = line.strip().split(' ')
    try:
        path = IMAGES_INPUT_FOLDER + line[1] + '.jpg'
        image = load_image(path)
        if image is not None:
            vote_sum = 0
            number_of_voters = 0
            for i in range(2, 11):
                vote_sum = vote_sum + (i - 1) * int(line[i])
                number_of_voters = number_of_voters + int(line[i])
            label = 0

            feature = {'train/label': _int64_feature(label),
                    'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))}

            example = tf.train.Example(features = tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
    except:
        pass

writer.close()
sys.stdout.flush()
