import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

from load_mot import LoadCOCO
import os

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def main(_argv):

    load = LoadCOCO('/home/brian/Desktop/AP/2DMOT2015/train/ADL-Rundle-6')

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    '''
    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)
    '''

    frame = 0
    dets = []
    
    assert (not load.empty())
    while not load.empty():
    
        path, img_raw, img = load.pop()
        name = os.path.basename(path)
        print (path)
        
        # nums = total detections.
        boxes, scores, classes, nums = yolo(img)
        
        nums_np = nums.numpy()
        num = nums_np[0]
        
        frames = frame * np.ones(shape=(num, 1))
        null = -1 * np.ones(shape=(num, 1))
        boxes_np = boxes.numpy()[0][0:num].reshape(num, 4)
        scores_np = scores.numpy()[0][0:num].reshape(num, 1)
        
        # print (np.shape(boxes_np))
        boxes_np[:, 0] = boxes_np[:, 0] * 1920
        boxes_np[:, 1] = boxes_np[:, 1] * 1080
        boxes_np[:, 2] = boxes_np[:, 2] * 1920
        boxes_np[:, 3] = boxes_np[:, 3] * 1080
        boxes_np[:, 2] = boxes_np[:, 2] - boxes_np[:, 0]
        boxes_np[:, 3] = boxes_np[:, 3] - boxes_np[:, 1]
        
        '''
        if len(boxes_np):
            print (boxes_np[0])
        '''
        
        det = np.concatenate((frames, null, boxes_np, scores_np, null, null, null), axis=1)
        dets.append(det)

        '''
        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])], np.array(scores[0][i]), np.array(boxes[0][i])))
        '''

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(name, img)
        
        frame = frame + 1

    #############################
    
    dets = np.concatenate(dets, axis=0)
    # print (np.shape(dets))
    
    # np.save('dets', dets)
    # np.savetxt("yolo-det.txt", dets, delimiter=",")
    np.savetxt("yolo-det.txt", dets, fmt='%d, %d, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %d, %d, %d', delimiter=",")
    # assert (False)
    
    #############################

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
