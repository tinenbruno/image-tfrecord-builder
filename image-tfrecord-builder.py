import numpy as np
import tensorflow as tf
import sys
import cv2
import os
from settings import app

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

writer = tf.python_io.TFRecordWriter(app['OUTPUT_FILENAME'])

for classname in os.listdir(app['IMAGES_INPUT_FOLDER']):
    class_dir = os.path.join(app['IMAGES_INPUT_FOLDER'], classname)
    if (os.path.isdir(class_dir)):
        for filename in os.listdir(class_dir):
            try:
                filepath = os.path.join(class_dir, filename)
                image = load_image(filepath)
                if image is not None:
                    feature = {
                        'train/label': _int64_feature(classname),
                        'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))
                    }

                    example = tf.train.Example(features = tf.train.Features(feature=feature))

                    writer.write(example.SerializeToString())
            except:
                pass

writer.close()
sys.stdout.flush()
