#!/usr/bin/env python
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# Updated by Kazuhiro Terao
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import sys,os
lib_path = os.path.join(os.environ['RCNNDIR'],'lib')
if not lib_path in sys.path:
    sys.path.insert(0,lib_path)

from config import cfg, cfg_from_file
for argv in sys.argv:
    if argv.endswith('.yml'): cfg_from_file(argv)
from rcnn_utils.nms_wrapper import nms
from rcnn_utils.test import im_detect
from rcnn_utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from vgg_faster_rcnn import vgg

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    #print('len(inds)={:d}'.format(len(inds)))
    #print('{:s}'.format(dets))
    im = im[:, :, (2, 1, 0)]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.35

    fig, ax = plt.subplots(figsize=(12, 12))
    png=ax.imshow(im[:,:,(2,1,0)], aspect='equal')
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #print('len(dets)={:d}'.format(len(dets)))
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #print('len(dets)={:d}'.format(len(dets)))
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig('out_%s.png' % (image_name))

if __name__ == '__main__':

    if not len(sys.argv) == 3:
        print('Error... usage: {:s} TF_CHECKPOINT'.format(sys.argv[0]))
        sys.exit(1)

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # model path
    #dataset = 'voc_2007_trainval'
    tfmodel = sys.argv[1]
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    net = vgg()
    net.create_architecture(len(CLASSES), mode="TEST", tag='default')
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg', 'vader.jpg', 'buckethead.jpg']
    im_names = ['vader.jpg','buckethead.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()
