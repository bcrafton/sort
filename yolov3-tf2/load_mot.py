
import os
import numpy as np
import cv2
import queue
import threading
import tensorflow as tf
from yolov3_tf2.dataset import transform_images

#########################################

def get_images(path):
    images = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if 'jpg' in file:
                full_path = os.path.join(subdir, file)
                if (full_path not in images):
                    images.append(full_path)

    return images

#########################################

def preprocess(filename):
    # image
    image_raw = tf.image.decode_image(open(filename, 'rb').read(), channels=3)
    image = tf.expand_dims(image_raw, 0)
    # image = transform_images(image, FLAGS.size)
    image = transform_images(image, 416)
    return image_raw, image

#########################################

def fill_queue(images, q):
    ii = 0
    last = len(images)

    while ii < last:
        if not q.full():
            filename = images[ii]
            image_raw, image = preprocess(filename)
            q.put((filename, image_raw, image))
            ii = ii + 1

#########################################

class LoadCOCO:

    def __init__(self, path):
        self.path = path
        self.images = sorted(get_images(self.path))
        self.q = queue.Queue(maxsize=1024)
        thread = threading.Thread(target=fill_queue, args=(self.images, self.q))
        thread.start()

    def pop(self):
        assert(not self.empty())
        return self.q.get()

    def empty(self):
        return self.q.qsize() == 0

    def full(self):
        return self.q.full()

###################################################################


















