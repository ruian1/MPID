import numpy as np
from larcv import larcv
import random

import ROOT
from ROOT import TChain

import torch
from torch.utils.data import Dataset

def image_modify(img):
    img_mod = np.where(img<10,    0,img)
    img_mod = np.where(img>500, 500,img_mod)
    return img_mod

class MPID_Dataset(Dataset):
    def __init__(self, input_file, mctruth_tree, image_tree, device, plane=2,augment=False, verbose=False):
        self.plane=plane
        self.augment=augment
        self.verbose=verbose
        self.particle_mctruth_chain = TChain(mctruth_tree)
        self.particle_mctruth_chain.AddFile(input_file)

        self.particle_image_chain = TChain(image_tree)
        self.particle_image_chain.AddFile(input_file)
        if (device):
            self.device=device
        else:
            self.device="cpu"
        
    def __getitem__(self, ENTRY):
        # Reading Image

        #print ("open ENTRY @ {}".format(ENTRY))

        self.particle_image_chain.GetEntry(ENTRY)
        self.this_image_cpp_object = self.particle_image_chain.sparse2d_wire_branch
        self.this_image=larcv.as_ndarray(self.this_image_cpp_object.as_vector()[self.plane])
        # Image Thresholding
        self.this_image=image_modify(self.this_image)

        #print (self.this_image)
        #print ("sum, ")
        #if (np.sum(self.this_image) < 9000):
        #    ENTRY+
        
        if self.augment:
            if random.randint(0, 1):
            #if True:
                #if (self.verbose): print ("flipped")
                self.this_image = np.fliplr(self.this_image)
            if random.randint(0, 1):
            #if True:
                #if (self.verbose): print ("transposed")
                self.this_image = self.this_image.transpose(1,0)
        self.this_image = torch.from_numpy(self.this_image.copy())
#        self.this_image=torch.tensor(self.this_image, device=self.device).float()

        self.this_image=self.this_image.clone().detach()
        
        # Reading Truth Info
        self.particle_mctruth_chain.GetEntry(ENTRY)
        self.this_mctruth_cpp_object = self.particle_mctruth_chain.particle_mctruth_branch
        self.this_mctruth = torch.zeros([5])

        for particle in self.this_mctruth_cpp_object.as_vector():
            if (particle.pdg_code()==11):
                self.this_mctruth[0]=1
            if (particle.pdg_code()==22):
                self.this_mctruth[1]=1
            if (particle.pdg_code()==13):
                self.this_mctruth[2]=1
            if (particle.pdg_code()==211 or particle.pdg_code()==-211):
                self.this_mctruth[3]=1
            if (particle.pdg_code()==2212):
                self.this_mctruth[4]=1
                                
        return (self.this_image, self.this_mctruth)

    def __len__(self):
        assert self.particle_image_chain.GetEntries()== self.particle_mctruth_chain.GetEntries()
        return self.particle_image_chain.GetEntries()

