import os
import time
import cv2
import numpy as np
import tensorflow as tf
from ctpn2.nets import model_train as model
from ctpn2.utils.rpn_msr.proposal_layer import proposal_layer
from ctpn2.utils.text_connector.detectors import TextDetector


class CTPN_model2:
    def __init__(self, checkpoint_path):
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            with self.sess.as_default():
                self.input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
                self.input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

                self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                                   trainable=False)

                self.bbox_pred, self.cls_pred, self.cls_prob = model.model(self.input_image)
                self.variable_averages = tf.train.ExponentialMovingAverage(0.997, self.global_step)
                self.saver = tf.train.Saver(self.variable_averages.variables_to_restore())
                # checkpoint_path = 'checkpoints_mlt/'
                self.checkpoint_path = checkpoint_path
                self.ckpt_state = tf.train.get_checkpoint_state(self.checkpoint_path)
                self.model_path = os.path.join(self.checkpoint_path,
                                               os.path.basename(self.ckpt_state.model_checkpoint_path))
                print('Restore from {}'.format(self.model_path))
                self.saver.restore(self.sess, self.model_path)
                self.textdetector = TextDetector(DETECT_MODE='H')

    def predict(self, img):
        start = time.time()
        # im = cv2.imread(img)[:, :, ::-1]
        #
        # img, (rh, rw) = self.resize_image(im)
        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        bbox_pred_val, cls_prob_val = self.sess.run([self.bbox_pred, self.cls_prob],
                                                    feed_dict={self.input_image: [img],
                                                               self.input_im_info: im_info})
        textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
        scores = textsegs[:, 0]
        textsegs = textsegs[:, 1:5]
        boxes = self.textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
        boxes = np.array(boxes, dtype=np.int)
        cost_time = (time.time() - start)
        print("cost time: {:.2f}s".format(cost_time))
        return boxes

    def resize_image(self, img):
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
            im_scale = float(1200) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)

        new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
        new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return re_im, (new_h / img_size[0], new_w / img_size[1])
