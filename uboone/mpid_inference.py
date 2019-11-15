from __future__ import division
from __future__ import print_function

import os, sys
from lib.config import config_loader
CFG = os.path.join("../cfg","inference_config.cfg")
cfg  = config_loader(CFG)

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="9"

print(sys.argv)
blah = sys.argv[1]

#test_file = "/scratch/ruian/training_data/MPID/larcv2/1e1p.root"
#test_file = "/scratch/ruian/training_data/MPID/larcv2/1mu1p.root"
#test_file = "/scratch/ruian/training_data/MPID/larcv2/all_1e1p.root"
test_file = "/scratch/ruian/training_data/MPID/larcv2/all_1mu1p.root"
#test_file = "/scratch/ruian/training_data/MPID/larcv2/test.root"
#test_file = "/scratch/ruian/training_data/MPID/larcv2/train.root"

blah+='_'
blah+=test_file.split('.')[0].split('/')[-1]

fout = open('inference_csvs/inference_{}.csv'.format(blah), 'w')
fout.write('entry,label0,label1,label2,label3,label4,score00,score01,score02,score03,score04,eng_ini0,eng_ini1,eng_ini2,eng_ini3,eng_ini4')
fout.write('\n')


from larcv import larcv
import ROOT
from ROOT import TChain

import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
#from torchvision import transforms
#from torchvision.datasets import MNIST

from mpid_data import mpid_data
from mpid_net import mpid_net, mpid_func


mctruth_tree="particle_mctruth_tree"
particle_mctruth_chain = TChain(mctruth_tree)
particle_mctruth_chain.AddFile(test_file)


train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test data

test_data = mpid_data.MPID_Dataset(test_file, "particle_mctruth_tree", "sparse2d_wire_tree", train_device)
test_loader = DataLoader(dataset=test_data, batch_size= 1 , shuffle=False)

mpid = mpid_net.MPID()
mpid.cuda()

#weight_file= "../weights/mpid_model_20191108-12_41_AM_epoch_29_batch_id_1831_title_LR-3_AG_True_new_modi_GN_changes_in_fullyCY_step_55974.pwf"

weight_file= "../weights/mpid_model_20191108-12_41_AM_epoch_29_batch_id_1811_title_LR-3_AG_True_new_modi_GN_changes_in_fullyCY_step_55954.pwf"

mpid.load_state_dict(torch.load(weight_file, map_location=train_device))

mpid.eval()

entry_start=5
#for ENTRY in xrange(entry_start, entry_start + 5):
for ENTRY in xrange(test_data.__len__()):

    truth_lable = test_data[ENTRY][1].cpu()

    input_image = test_data[ENTRY][0].view(-1,1,512,512)

    eng_ini = np.zeros(5)
    
    if (input_image.sum().cpu() < 100):
        score = torch.zeros([5])
    else:
        score = nn.Sigmoid()(mpid(input_image.cuda())).cpu().detach().numpy()[0]
        #print (score)
        
    fout.write("{},".format(ENTRY))
    for each in truth_lable:
        fout.write("{:d},".format(int(each)))
    for each in score:        
        fout.write("{:f},".format(float(each)))


    particle_mctruth_chain.GetEntry(ENTRY)
    mctruth_cpp_object = particle_mctruth_chain.particle_mctruth_branch
    for particle in mctruth_cpp_object.as_vector():
        if (particle.pdg_code()==11 and eng_ini[0]<particle.energy_init()):
            eng_ini[0]=particle.energy_init()
        if (particle.pdg_code()==22 and eng_ini[1]<particle.energy_init()):
            eng_ini[1]=particle.energy_init()
        if (particle.pdg_code()==13 and eng_ini[2]<particle.energy_init()):
            eng_ini[2]=particle.energy_init()
        if (particle.pdg_code()==211 or particle.pdg_code()==-211 and eng_ini[3]<particle.energy_init()):
            eng_ini[3]=particle.energy_init()
        if (particle.pdg_code()==2212 and eng_ini[4]<particle.energy_init()):
            eng_ini[4]=particle.energy_init()

    ctr = 0
    for each in eng_ini:        
        if ctr < 4 :
            fout.write("{:f},".format(float(each)))
        else:
            fout.write("{:f}".format(float(each)))
        ctr+=1
    fout.write('\n')
fout.close()
