


import os, sys
import numpy as np
from PIL import Image
import cv2
import shutil
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, sampler
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname


from Utils import biuld_train_loader, EdgeNet, cross_entropy_loss, Logger
from Utils import train, Averagvalue, save_checkpoint, load_vgg16pretrain


#  hyperparameters
lr=0.000001
weight_decay= 2e-4
momentum=0.9
stepsize=3
gamma=0.1
maxepoch=15
itersize=10 # defalut is 10

number_of_classes=2
batch_size =1
mask_thresh=128

# ____________________________________train loader
directory_train='.data/train'
directory_list='.data/list.lst'

train_loader= biuld_train_loader(directory_train,number_of_classes, batch_size,mask_thresh,directory_list)



#__________________ creat the model an initialize the weights
def weights_init(m):
  if isinstance(m,nn.Conv2d):
    m.weight.data.normal_(0, 0.01)
    if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
      torch.nn.init.constant_(m.weight, 0.2)
    if m.bias is not None:
      m.bias.data.zero_()

    
  # model
model = EdgeNet()
model.cuda()
model.apply(weights_init)
load_vgg16pretrain(model,'vgg16convs.mat')



net_parameters_id = {}
net = model
for pname, p in net.named_parameters():
    if pname in ['conv1_1.weight','conv1_2.weight',
                 'conv2_1.weight','conv2_2.weight',
                 'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                 'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
        print(pname, 'lr:1 de:1')
        if 'conv1-4.weight' not in net_parameters_id:
            net_parameters_id['conv1-4.weight'] = []
        net_parameters_id['conv1-4.weight'].append(p)
    elif pname in ['conv1_1.bias','conv1_2.bias',
                   'conv2_1.bias','conv2_2.bias',
                   'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                   'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
        print(pname, 'lr:2 de:0')
        if 'conv1-4.bias' not in net_parameters_id:
            net_parameters_id['conv1-4.bias'] = []
        net_parameters_id['conv1-4.bias'].append(p)
    elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
        print(pname, 'lr:100 de:1')
        if 'conv5.weight' not in net_parameters_id:
            net_parameters_id['conv5.weight'] = []
        net_parameters_id['conv5.weight'].append(p)
    elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias'] :
        print(pname, 'lr:200 de:0')
        if 'conv5.bias' not in net_parameters_id:
            net_parameters_id['conv5.bias'] = []
        net_parameters_id['conv5.bias'].append(p)
    elif pname in ['conv1_1_down.weight','conv1_2_down.weight',
                   'conv2_1_down.weight','conv2_2_down.weight',
                   'conv3_1_down.weight','conv3_2_down.weight','conv3_3_down.weight',
                   'conv4_1_down.weight','conv4_2_down.weight','conv4_3_down.weight',
                   'conv5_1_down.weight','conv5_2_down.weight','conv5_3_down.weight']:
        print(pname, 'lr:0.1 de:1')
        if 'conv_down_1-5.weight' not in net_parameters_id:
            net_parameters_id['conv_down_1-5.weight'] = []
        net_parameters_id['conv_down_1-5.weight'].append(p)
    elif pname in ['conv1_1_down.bias','conv1_2_down.bias',
                   'conv2_1_down.bias','conv2_2_down.bias',
                   'conv3_1_down.bias','conv3_2_down.bias','conv3_3_down.bias',
                   'conv4_1_down.bias','conv4_2_down.bias','conv4_3_down.bias',
                   'conv5_1_down.bias','conv5_2_down.bias','conv5_3_down.bias']:
        print(pname, 'lr:0.2 de:0')
        if 'conv_down_1-5.bias' not in net_parameters_id:
            net_parameters_id['conv_down_1-5.bias'] = []
        net_parameters_id['conv_down_1-5.bias'].append(p)
    elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                   'score_dsn4.weight','score_dsn5.weight']:
        print(pname, 'lr:0.01 de:1')
        if 'score_dsn_1-5.weight' not in net_parameters_id:
            net_parameters_id['score_dsn_1-5.weight'] = []
        net_parameters_id['score_dsn_1-5.weight'].append(p)
    elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                   'score_dsn4.bias','score_dsn5.bias']:
        print(pname, 'lr:0.02 de:0')
        if 'score_dsn_1-5.bias' not in net_parameters_id:
            net_parameters_id['score_dsn_1-5.bias'] = []
        net_parameters_id['score_dsn_1-5.bias'].append(p)
    elif pname in ['score_final.weight']:
        print(pname, 'lr:0.001 de:1')
        if 'score_final.weight' not in net_parameters_id:
            net_parameters_id['score_final.weight'] = []
        net_parameters_id['score_final.weight'].append(p)
    elif pname in ['score_final.bias']:
        print(pname, 'lr:0.002 de:0')
        if 'score_final.bias' not in net_parameters_id:
            net_parameters_id['score_final.bias'] = []
        net_parameters_id['score_final.bias'].append(p)

    elif pname in ['score_final_h.weight']: # hossein
        print(pname, 'lr:0.001 de:1')
        if 'score_final_h.weight' not in net_parameters_id:
            net_parameters_id['score_final_h.weight'] = []
        net_parameters_id['score_final_h.weight'].append(p)
    elif pname in ['score_final_h.bias']:
        print(pname, 'lr:0.002 de:0')
        if 'score_final_h.bias' not in net_parameters_id:
            net_parameters_id['score_final_h.bias'] = []
        net_parameters_id['score_final_h.bias'].append(p)
 ##_________________________ I might change it to adam or other optimizers      Hossein
optimizer = torch.optim.SGD([ 
        {'params': net_parameters_id['conv1-4.weight']      , 'lr': lr*1    , 'weight_decay': weight_decay},
        {'params': net_parameters_id['conv1-4.bias']        , 'lr': lr*2    , 'weight_decay': 0.},
        {'params': net_parameters_id['conv5.weight']        , 'lr': lr*100  , 'weight_decay': weight_decay},
        {'params': net_parameters_id['conv5.bias']          , 'lr': lr*200  , 'weight_decay': 0.},
        {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': lr*0.1  , 'weight_decay': weight_decay},
        {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': lr*0.2  , 'weight_decay': 0.},
        {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': lr*0.01 , 'weight_decay': weight_decay},
        {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': lr*0.02 , 'weight_decay': 0.},
        {'params': net_parameters_id['score_final.weight']  , 'lr': lr*0.001, 'weight_decay': weight_decay},
        {'params': net_parameters_id['score_final.bias']    , 'lr': lr*0.002, 'weight_decay': 0.},
        {'params': net_parameters_id['score_final_h.weight']  , 'lr': lr*0.001, 'weight_decay': weight_decay},
        {'params': net_parameters_id['score_final_h.bias']    , 'lr': lr*0.002, 'weight_decay': 0.},
    ], lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)




#___________________   Train the model

train_loss = []
train_loss_detail = []
for epoch in range(0, maxepoch):
    tr_avg_loss, tr_detail_loss = train(train_loader, model,
                                        optimizer, epoch,itersize,maxepoch,print_freq)

    #save_file = os.path.join(TMP_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
    #save_checkpoint({
        #'epoch': epoch,
        #'state_dict': model.state_dict(),
        #'optimizer': optimizer.state_dict()
        #             }, filename=save_file)
    scheduler.step() # will adjust learning rate
    # save train/val loss/accuracy, save every epoch in case of early stop
    train_loss.append(tr_avg_loss)
    train_loss_detail += tr_detail_loss 



torch.save(model,'EdgeNet1')