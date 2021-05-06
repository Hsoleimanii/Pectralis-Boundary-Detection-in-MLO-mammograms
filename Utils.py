



from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2
import os, sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _single, _pair, _triple
import torch.nn.functional as F
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

from os.path import join as pjoin
import skimage.io as io
import time
import skimage
import warnings



def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def prepare_image_cv2(im):
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im



        
def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))



from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class FormsDataset(Dataset):
    def __init__(self, dirr, mask_thresh=128, list_dir=None,transform=None,splitt='train'):
        self.transforms = transform
        self.dir=dirr
        self.mask_thresh=mask_thresh
        self.list_dir=list_dir
        self.split=splitt
        self.files =self.find_list(dirr,list_dir) 
        print(self.files[0])
        
    def __getitem__(self, idx):
        if self.split=='train':
            idx=idx*2
            #idx=16
            image = self.read_im(self.files[idx],cv2.IMREAD_COLOR) #IMREAD_COLOR
            if self.transforms:
                image = self.transforms(image)

            mask = self.read_mask(self.files[idx+1],cv2.IMREAD_GRAYSCALE)    
            return image, mask
        else:
            image = self.read_im(self.files[idx],cv2.IMREAD_COLOR) #IMREAD_COLOR
            if self.transforms:
                image = self.transforms(image)
            return image
    
    def __len__(self):
        if self.split=='train':
            return  (np.array(len(self.files)/2)).astype(np.int)  
        else:
            return (np.array(len(self.files)/1)).astype(np.int)
 
    def read_im(self,path, mod):
        tmp = cv2.imread(path, mod)
        tmp = np.array(tmp).astype(np.float32)
        
        #im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr  
        tmp -= np.array((104.00698793,116.66876762,122.67891434))
        tmp = np.transpose(tmp, (2, 0, 1)) # (H x W x C) to (C x H x W)
        #tmp=tmp.reshape(1,tmp.shape[0],tmp.shape[1]) # for gray image
        return tmp
    
    def read_mask(self,path, mod):
        tmp = cv2.imread(path, mod)
        tmp = np.array(tmp).astype(np.float32)  
        tmp[np.where(tmp<self.mask_thresh)]=0
        tmp[np.where(tmp>=self.mask_thresh)]=1
        tmp=tmp.reshape(1,tmp.shape[0],tmp.shape[1])
        return tmp
    
    def find_list(self,dirr,list_dir):
        with open(list_dir, 'r') as f:
            filelist =f.read().splitlines()
        
        ff=[]
        for f in filelist:
            ff.append(join(dirr,str(f)))
        return ff
    

def biuld_train_loader(directory_train,number_of_classes, batch_size,mask_thresh,list_dir):
    #train_x, test_x, train_y, test_y=read_data(directory_train,directory_test)
    train_dataset = FormsDataset(directory_train, mask_thresh,list_dir, splitt="train")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f'Train dataset has {len(train_data_loader)} batches of size {batch_size}')
    return train_data_loader

def biuld_test_loader(directory_test,number_of_classes, batch_size):
    test_dataset = FormsDataset(directory_test,list_dir ,splitt='test')
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f'Test dataset has {len(test_data_loader)} batches of size {batch_size}')
    
    return  test_data_loader


# loss function

def cross_entropy_loss_(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

# note that \lambda is equal to 0.1
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)



###############  MODEL___________________________________________________________________________________________





class EdgeNet(nn.Module):
    def __init__(self):
        super(EdgeNet, self).__init__()
        #lr 1 2 decay 1 0
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1) # input channels is 1  grayscale images
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)


        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                        stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                        stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                        stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)


        #lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 11, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 11, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(512, 11, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(512, 11, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(512, 11, 1, padding=0)
        
        self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(512, 21, 1, padding=0)

        #lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)



    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1   = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2   = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3   = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4   = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)
        ## transpose and crop way 
        weight_deconv2 =  make_bilinear_weights(4, 1).cuda()
        weight_deconv3 =  make_bilinear_weights(8, 1).cuda()
        weight_deconv4 =  make_bilinear_weights(16, 1).cuda()
        weight_deconv5 =  make_bilinear_weights(32, 1).cuda()

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)
        
        ### center crop
        #so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        #so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        #so5 = crop(upsample5, img_H, img_W)



        fuse = self.score_final(fusecat)
        fuse_h=self.score_final_h(fuse); 
        results =[so2, so3]



        results = [torch.sigmoid(r) for r in results]
        return results


def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]


def crop_caffe(location, variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(location)
        y1 = int(location)
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]

# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w

def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)




def train(train_loader, model, optimizer, epoch,itersize,maxepoch,print_freq):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        outputs = model(image)

        
        loss = torch.zeros(1).cuda()
        #print(len(outputs))
        #print(outputs[0].detach().cpu().numpy().shape)
        #for o in outputs:
        #   loss = loss + cross_entropy_loss_RCF(o, label)

        loss = loss + cross_entropy_loss_RCF(outputs[0], label)+ cross_entropy_loss_RCF(outputs[1], label)  ##### hossein

        optimizer.zero_grad() # reset gradient ????
        loss.backward()    # backward-pass
        optimizer.step()      # update weights using a reasonably sized gradient ????


        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            #print(len(outputs))
            outputs.append(label)
            #print(len(outputs))
           # _, _, H, W = outputs[0].shape
           # all_results = torch.zeros((len(outputs), 1, H, W))
           # for j in range(len(outputs)):
            #    all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
           # torchvision.utils.save_image(all_results, join(save_dir, "iter-%d.jpg" % i))

        # save checkpoint
    ##########save_checkpoint({
      #  'epoch': epoch,
       # 'state_dict': model.state_dict(),
        #'optimizer': optimizer.state_dict()
         #   }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))

    return losses.avg, epoch_loss




class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()



def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))

def load_vgg16pretrain(model, vggmodel='..\input\vgg_16\vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)




def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1,4,1,1]):
            torch.nn.init.constant_(m.weight, 0.25)
        if m.bias is not None:
            m.bias.data.zero_()