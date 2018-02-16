# Example script to run with toydata particles datasets
import sys,os

lib_path = os.path.join(os.environ['RCNNDIR'], 'lib')
if not lib_path in sys.path:
    sys.path.insert(0, lib_path)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import ROOT
from larcv import larcv
import tensorflow as tf

from rcnn_train.trainer import train_net
from toydata_generator import ToydataGenerator
from vgg_faster_rcnn import vgg

net = vgg()
train_io = ToydataGenerator(256, 2, 3)
val_io = ToydataGenerator(256, 2, 3)

print "Setting input shape"
image = tf.zeros(shape=[1,256,256,1], dtype=tf.int32)
net.set_input_shape(image)

print "Starting training"
train_net(net,
        os.path.join(os.environ['RCNNDIR'], 'output/toydata0'),
        os.path.join(os.environ['RCNNDIR'], 'tensorboard/toydata0'),
        train_io,
        val_io,
        '%s/data/vgg16.ckpt' % os.environ['RCNNDIR'])
