import tensorflow as tf
import numpy as np
import cv2
from .cfg import CTPNConfig
from lib.fast_rcnn.config import cfg
from lib.networks.factory import get_network
from .text_proposal import TextDetector
from lib.utils.blob import im_list_to_blob
from .other import draw_boxes


class CTPN_model:
    def __init__(self, ckpt_path):
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.net = get_network("VGGnet_test")
                self.saver = tf.train.Saver()
                self.ckpt = tf.train.get_checkpoint_state(ckpt_path)
                self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
                self.scale, self.max_scale = CTPNConfig.SCALE, CTPNConfig.MAX_SCALE
                self.textdetector = TextDetector()

    def predict(self, img):
        blobs, im_scales = self._get_blobs(img)
        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        feed_dict = {
            self.net.data: blobs['data'],
            self.net.im_info: blobs['im_info'],
            self.net.keep_prob: 1.0
        }
        rois = self.sess.run([self.net.get_output('rois')[0]], feed_dict=feed_dict)
        rois = rois[0]
        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        boxes = self.textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        text_recs, tmp = draw_boxes(img, boxes, caption='im_name', wait=True, is_display=True)
        box = sorted(text_recs, key=lambda x: sum([x[1], x[3], x[5], x[7]]))

        return box

    def _get_image_blob(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(
                im_orig,
                None,
                None,
                fx=im_scale,
                fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def _get_blobs(self, im):
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scale_factors = self._get_image_blob(im)
        return blobs, im_scale_factors
