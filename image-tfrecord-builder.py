import os
import sys

import cv2
import numpy as np

import tensorflow as tf
from settings import app


def _load_image(path):
    image = cv2.imread(path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32)
    return None

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _build_examples_list():
    examples = []
    for classname in os.listdir(app['IMAGES_INPUT_FOLDER']):
        class_dir = os.path.join(app['IMAGES_INPUT_FOLDER'], classname)
        if (os.path.isdir(class_dir)):
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                example = {
                    'classname': classname, 
                    'path': filepath
                }
                examples.append(example)

    return examples

def _split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def _get_examples_share(examples, training_split):
    examples_size = len(examples)
    len_training_examples = int(examples_size * training_split)

    return np.split(examples, [len_training_examples])

def _write_tfrecord(examples, output_filename):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for example in examples:
        try:
            image = _load_image(example.path)
            if image is not None:
                feature = {
                    'train/label': _int64_feature(example.classname),
                    'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))
                }

                tf_example = tf.train.Example(features = tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())
        except:
            pass
    writer.close()

examples = _build_examples_list()
training_examples, test_examples = _get_examples_share(examples, app['TRAINING_EXAMPLES_SPLIT']) # pylint: disable=unbalanced-tuple-unpacking

_write_tfrecord(training_examples, app['OUTPUT_FILENAME'] + '.training')
_write_tfrecord(test_examples, app['OUTPUT_FILENAME'] + '.test')

sys.stdout.flush()
