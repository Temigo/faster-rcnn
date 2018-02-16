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
from rcnn_utils.test import im_detect, _clip_boxes
from rcnn_utils.bbox_transform import bbox_transform_inv
from rcnn_utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from vgg_faster_rcnn import vgg
from toydata_generator import ToydataGenerator

CLASSES = ('__background__', 'track_edge', 'shower_start')

def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    print(dets.shape)
    print(dets[:, -1])
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

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

def inference(sess, net, index=-1):
    train_io = ToydataGenerator(256,2,3)
    blob = train_io.forward()

    im_scale = float(cfg.TEST.SCALE) / float(min(blob['im_info'][1], blob['im_info'][2]))
    if np.round(im_scale * max(blob['im_info'][1], blob['im_info'][2])) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(max(blob['im_info'][1], blob['im_info'][2]))

    _, scores, bbox_pred, rois = net.test_image(sess, blob['data'], blob['im_info'])

    boxes = rois[:, 1:5] / im_scale
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, (blob['im_info'][1], blob['im_info'][2]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes, blob

def demo(sess, net):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'particles', image_name)
    #im = cv2.imread(im_file)
    #print(im.shape)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    #scores, boxes = im_detect(sess, net, im)
    scores, boxes, blob = inference(sess, net)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1 #0.8
    NMS_THRESH = 0.0 #0.35

    im = blob['data'][0,:,:,0]
    fig, ax = plt.subplots(figsize=(12,12), facecolor='w')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #png=ax.imshow(im[:,:,(2,1,0)], interpolation='none', cmap='jet', origin='lower')
    png=ax.imshow(im[:,:], interpolation='none', cmap='jet', origin='lower')
    plt.axis('off')
    plt.tight_layout()

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


    plt.draw()
    fig.savefig('out_toydata.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #if not len(sys.argv) == 3:
    #    print('Error... usage: {:s} TF_CHECKPOINT'.format(sys.argv[0]))
    #    sys.exit(1)

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
    #im_names = ['image0.png','image1.png', 'image2.png', 'image50.png', 'image100.png', 'image500.png']
    #for im_name in im_names:
    #    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #    print('Demo for data/particles/{}'.format(im_name))
    #    demo(sess, net, im_name)
    demo(sess, net)
    plt.show()
