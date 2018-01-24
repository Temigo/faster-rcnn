#---------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under the MIT License [see LICENSE for details]
# --------------------------------------------------------

import sys,os
import numpy as np
import ROOT
#from ROOT import TChain
from larcv import larcv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class particlesdata_gen(object):

    CLASSES = ('__background__',
    'electron', 'muon',
    'proton', 'gammaray',
    'chargedpion')

    def __init__(self, filename='train_15k.root', debug=False):
        if debug: print "Init"
        self.filename = os.path.join(os.environ['RCNNDIR'], 'data/' + filename)
        if debug: print "\nLoading " + self.filename + "...\n"
        self.chain_image2d = ROOT.TChain('image2d_data_tree')
        self.chain_image2d.AddFile(self.filename)
        self.chain_particle_mcst = ROOT.TChain('particle_mcst_tree')
        self.chain_particle_mcst.AddFile(self.filename)
        if debug: print "Done.\n"
        self.index = -1
        self._debug = debug

    def num_classes(self):
        return 6

    def make_data(self, index=-1):
        self.chain_image2d.GetEntry(index)
        self.chain_particle_mcst.GetEntry(index)
        #print self.chain_particle_mcst.branchnames
        #if self._debug:
        print "Index %d" % index
        particles = self.chain_particle_mcst.particle_mcst_branch.as_vector()

        # Using only first of 3 (projections) images in each entry
        img = larcv.as_ndarray(self.chain_image2d.image2d_data_branch.as_vector().front())
        fig, ax = plt.subplots(figsize=(12,8), facecolor='w')
        ax.imshow(img, interpolation='none', cmap='jet', origin='lower')
        fig.savefig('image%d.png' % index)

        labels = []
        for particle in particles:
            box = particle.boundingbox_2d().front()

            labels.append([box.min_x(),
            box.min_y(),
            box.min_x() + box.width(),
            box.min_y() + box.height(),
            particle.pdg_code()])

        return img, np.array(labels)

    def forward(self, index=-1):
        #self.index += 1
        self.index = 2
        img, labels = self.make_data(self.index)

        img = img.reshape(256, 256)
        img = img[np.newaxis,:,:,np.newaxis]
        #print img.shape
        img = np.repeat(img, 3, axis=3)
        #print img.shape
        #img[:,:,:,1] = np.zeros((img.shape[0],img.shape[1], img.shape[2]))
        #img[:,:,:,2] = np.zeros((img.shape[0],img.shape[1], img.shape[2]))
        #print img[0,:,:,1].shape
        if self._debug: print img.shape

        blob = {}
        blob['data'] = img
        blob['im_info'] = np.array([1, img.shape[1], img.shape[2],3])
        blob['gt_boxes'] = labels


        return blob

if __name__ == '__main__':
    g = particlesdata_gen('/home/ldomine/train_15k.root')
    blob = g.forward(index=9)
    print blob['data'].shape
    print blob['data']
    print blob['im_info']
    print blob['gt_boxes']
