# Example script to run with particles datasets
#from __future__ import absolute_import
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
from rcnn_train.particlesdata import particlesdata_gen
from vgg_faster_rcnn import vgg

net = vgg()
train_io = particlesdata_gen(filename='test_10k.root', debug=False)
val_io = particlesdata_gen(filename='test_10k.root', debug=False)

train_io.forward()

print "Setting input shape"
image = tf.zeros(shape=[1,256,256,3], dtype=tf.float32)
net.set_input_shape(image)

print "Starting training"
train_net(net, 'output/particle', 'tensorboard/particle', train_io, val_io, '%s/data/vgg16.ckpt' % os.environ['RCNNDIR'])
