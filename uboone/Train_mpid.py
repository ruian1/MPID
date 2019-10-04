from __future__ import division
from __future__ import print_function

import os, sys
from lib.config import config_loader
from lib.utility import timestr

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
CFG = os.path.join(BASE_PATH,"../cfg","simple_config.cfg")
cfg  = config_loader(CFG)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from mpid_data import mpid_data
from mpid_net import mpid_net, mpid_func

SEED = 1

cuda = torch.cuda.is_available()

# # For reproducibility
# torch.manual_seed(SEED)

# if cuda:
#     torch.cuda.manual_seed(SEED)

print ("Using {} GPUs".format(torch.cuda.device_count()))


train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#test_device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


# Training data
train_file = "/scratch/ruian/training_data/MPID/larcv2/train.root"
train_data = mpid_data.MPID_Dataset(train_file, "particle_mctruth_tree", "sparse2d_wire_tree", train_device)
train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size_train, shuffle=True)

# Test data
test_file = "/scratch/ruian/training_data/MPID/larcv2/test.root"
test_data = mpid_data.MPID_Dataset(test_file, "particle_mctruth_tree", "sparse2d_wire_tree", train_device)
test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size_test, shuffle=False)

mpid = mpid_net.MPID()
mpid.cuda()

# BCEWithLogitsLoss seems not have sigmoid, ticket submited on PyTorch forum
# Using Sigmoid in mpidnet + BCELoss instead 
loss_fn = nn.BCEWithLogitsLoss()
#loss_fn = nn.BCELoss()

optimizer = optim.Adam(mpid.parameters(), lr=1e-4, weight_decay=0.001)
train_step = mpid_func.make_train_step(mpid, loss_fn, optimizer, trainable = True)
test_step = mpid_func.make_train_step(mpid, loss_fn, optimizer, trainable = False)

print ("Training with {} images".format(len(train_loader.dataset)))

train_losses = []
train_accuracies =[]
test_losses = []
test_accuracies =[]

EPOCHS = cfg.EPOCHS

fout = open('production_{}.csv'.format(timestr()), 'w')
fout.write('train_accu,test_accu,train_loss,test_loss')
fout.write('\n')

for epoch in range(EPOCHS):
    print ("@{} epoch...".format(epoch))
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        # the dataset "lives" in the CPU, so do our mini-batches
        # therefore, we need to send those mini-batches to the
        # device where the model "lives"

        print ("@ epoch {}, @ batch_id {}".format(epoch+1, batch_idx))
        
        x_batch = x_batch.to(train_device).view((-1,1,512,512))
        y_batch = y_batch.to(train_device)
                
        loss = train_step(x_batch, y_batch, trainable=True) #model.train() called in train_step
        train_losses.append(loss)

            
        print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch+1,
            EPOCHS,
            batch_idx * len(x_batch), 
            len(train_loader.dataset),
            100. * batch_idx / len(train_loader), 
            loss), 
            end='')

        
        if (batch_idx % 100 == 1 and cfg.run_test):
            torch.save(mpid.state_dict(), "/scratch/ruian/MPID_pytorch/weights/mpid_model_{}_{}_{}.pwf".format(timestr(), epoch+1, batch_idx))

            print ("Start eval on test sample...@..{}@..{}".format(epoch+1, batch_idx))
            test_accuracy = mpid_func.validation(mpid, test_loader, cfg.batch_size_test, train_device, event_nums=1280)
            print ("Test Accuray {}".format(test_accuracy))
            print ("Start eval on training sample...@..{}@..{}".format(epoch+1, batch_idx))
            train_accuracy = mpid_func.validation(mpid, train_loader, cfg.batch_size_train, train_device, event_nums=1280)
            print ("Train Accuray {}".format(train_accuracy))
            test_loss= test_step(x_batch, y_batch, trainable=False)
            fout.write("%f,"%train_accuracy)        
            fout.write("%f,"%test_accuracy)
            fout.write("%f,"%loss)
            fout.write("%f"%test_loss)
            fout.write("\n")
            
fout.close()

        
