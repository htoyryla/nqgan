from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
#from model3blx import _netG, _netD, _netE, weights_init, TVLoss
from torch.optim import lr_scheduler
import numpy as np
import cv2
#import torch.nn.functional as F
#from DiscriminativeLRt import discriminative_lr_params
from FocalLoss import WeightedFocalLoss 
from image_pool import ImagePool

# gan trainer
# htoyryla 30 Jul 2018
#
# put images in datasets/<name>/train/
#

# older history
# v.2c3 adds noisy labels
# v.2c4 add loading of netE weights
# v.2c5 add non-strict (partial) loading of pretrained weights (requires model2c3)
# v.2c6 add optional freezing of already trained layers
# v.2c7 use noisy labels only in discriminator, models: use larger kernels in upper layers 
# v.2c8 models changed so that in D & E, layers are added to input size as image size grows
# v.2c8n adding noise to input 
# v.2c9 relativistic training, deeper discriminator (model2c6)
# v.2c9b encoder training corrected, sigmoid removed etc

# new start 30 Jul 2018
# hsv not supported currently
# model will be be rewritten and use different architecture
# progressive training not to be supported (models trained with smaller image size will not be extensible to larger size)

# haegan used image -> embedding -> image' to train encoder (instead of emb -> image -> emb' as in hgan)
# v3b merge haegan into hgan, new option how to train netE, uses model3b with refactored model creation
# 2i2x uses default initialization, leakyrelus in G, superres layer in G

# merged various things into a new start hxgan
parser = argparse.ArgumentParser()
parser.add_argument('--runroot', default='./runs', help='path to where the project folders are stored')
parser.add_argument('--dataset', default='folder', required=True, help=' folder | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--dsmeta', default="", help='path to dataset metadata')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')

parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nef', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--lmbd', type=float, default=100, help='lambda, default=100')
parser.add_argument('--tvloss', type=float, default=0.0002, help='tv loss, default=0.0002')
parser.add_argument('--niter', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--save_every', type=int, default=10, help='number of epochs between saves')
parser.add_argument('--save_everyD', type=int, default=-1, help='number of epochs between saves')
parser.add_argument('--imgStep', type=int, default=0, help='minibatches between image folder saves')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning G rate, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning D rate, default=0.0002')
parser.add_argument('--lrE', type=float, default=0.0002, help='learning E rate, default=0.0002')
parser.add_argument('--cyclr', type=float, default=-1, help='')
parser.add_argument('--step', type=int, default=40, help='lr step, default=40')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma, default=0.1')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--bneps', type=float, default=1e-5, help='eps for batch norm')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--name', default='baseline', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0 0,1,2 0,2')
parser.add_argument('--lsgan', action='store_true', help='use lsgan')
parser.add_argument('--hgan', action='store_true', help='use hgan style discriminator')
parser.add_argument('--hgan2', action='store_true', help='use hgan style 2 discriminator')
parser.add_argument('--rsgan', action='store_true', help='use rsgan')
parser.add_argument('--D2', action='store_true', help='use additional convlayers in D')
parser.add_argument('--elu', action='store_true', help='use ELU')
parser.add_argument('--sinerelu', action='store_true', help='use SineReLU')
parser.add_argument('--instance', action='store_true', help='use instance norm')
parser.add_argument('--batchnorm', action='store_true', help='use batch norm')
parser.add_argument('--nonorm', action='store_true', help='no norm layer')
parser.add_argument('--upsample', action='store_true', help='use upsample and conv instead of convtranspose')
parser.add_argument('--upsample_after', type=int, default=1000, help='use upsample on top layers')
parser.add_argument('--cudnn', action='store_true', help='use cudnn')
parser.add_argument('--benchmark', action='store_true', help='use cudnn')
parser.add_argument('--determ', action='store_true', help='use cudnn')
parser.add_argument('--withoutE', action='store_true', help='do not use Encoder Network')
parser.add_argument('--debug', action='store_true', help='show debug info')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight. Greatly helps convergence but leads to artifacts in images, not recommended.')
parser.add_argument('--l1reg', type=float, default=0, help='L1 regularization weight.')
parser.add_argument('--nlabels', action='store_true', help='use noisy labels')
parser.add_argument('--nostrict', action='store_true', help='allow partial loading of pretrained nets')
parser.add_argument('--noise', type=float, default=0.0, help='input noise, default=0.')
parser.add_argument('--flip', action='store_true', help='random horizontal flip')
parser.add_argument('--crop', action='store_true', help='center crop into square')
parser.add_argument('--randomcrop', action='store_true', help='random crop into square')
parser.add_argument('--trainE', default="Z", help='how encoder is used')
parser.add_argument('--init', default="default", help='default | dcgan')
parser.add_argument('--nolin', action='store_true', help='use some linear layers in models')
parser.add_argument('--clipD', type=float, default=0, help='clip discriminator weights to given value')
parser.add_argument('--clipG', type=float, default=0, help='clip generator weights to given value')
parser.add_argument('--clipE', type=float, default=0, help='clip generator weights to given value')
parser.add_argument('--Dsize', type=float, default=0, help='image size for discriminator for special effects')
parser.add_argument('--flip_labels', action='store_true', help='flip real and fake labels')
parser.add_argument('--median', type=int, default=0, help='median layer kernel, set 0 for no median filter')
parser.add_argument('--medianindex', type=str, default="", help='')
parser.add_argument('--medianstride', type=int, default=1, help='')
parser.add_argument('--dmedian', type=int, default=0, help='median layer kernel, set 0 for no median filter')
parser.add_argument('--dmedianindex', type=str, default="", help='')
parser.add_argument('--medianout', type=int, default=0, help='')
parser.add_argument('--dmedianin', type=int, default=0, help='')
parser.add_argument('--minibatchstd', action='store_true', help='use minibatch std in D')
parser.add_argument('--saveGextra', action='store_true', help='save G always when saving sample images')
parser.add_argument('--x2', action='store_true', help='use additional layer for double size')
parser.add_argument('--x4', action='store_true', help='use two additional layers for 4x size')
parser.add_argument('--reallabel', type=float, default=1.0, help='')
parser.add_argument('--fakelabel', type=float, default=0.0, help='')
parser.add_argument('--freezeD', type=str, default="", help='freeze given layers in D')
parser.add_argument('--freezeG', type=str, default="", help='freeze given layers in G')
parser.add_argument('--blur', type=int, default=0, help='initial blur, use 0 for no blur')
parser.add_argument('--blurstep', type=int, default=4, help='number of epochs until stepping down blur')
parser.add_argument('--blurdelta', type=int, default=2, help='how much blur is decreased at each step')
parser.add_argument('--blurnoise', type=float, default=0.0, help='noise to be added after blur, default=0.')
parser.add_argument('--blurmode', type=str, default="blur", help='blur mode blur/alt')
parser.add_argument('--blurtype', type=str, default="box", help='blur type box/median')
parser.add_argument('--fixednoise', action='store_true', help='use fixed noise when saving sample images')
#parser.add_argument('--laplacian', type=int, default= 0, help='laplacian kernel size, 0 for no laplacian filter')
parser.add_argument('--dropout', type=float, default=0, help='dropout prob for selected layers in D')
parser.add_argument('--dropoutindex', type=str, default="", help='D layers to add dropout')
parser.add_argument('--brightness', type=float, default=0, help='modify dataset brightness')
parser.add_argument('--contrast', type=float, default=0, help='modify dataset contrast')
parser.add_argument('--saturation', type=float, default=0, help='modify dataset saturation')
parser.add_argument('--drandom', action='store_true', help='random dataset modification')
parser.add_argument('--orthoD', action='store_true', help='orthogonal D weights')
parser.add_argument('--orthoG', action='store_true', help='orthogonal G weights')
parser.add_argument('--hardtanh', action='store_true', help='use hardtang in G')
parser.add_argument('--dlr', type=str, default='', help='use discriminative lr')
parser.add_argument('--dlrrange', type=int, default=0, help='use discriminative lr')
#parser.add_argument('--dlrGrev', action='store_true', help='reverse lr scales for G')
#parser.add_argument('--dlrDrev', action='store_true', help='reverse lr scales for D')
parser.add_argument('--residual', action='store_true', help='use residual in G')
parser.add_argument('--attention', type=str, default="", help='')
parser.add_argument('--attentionD', type=str, default="", help='')
parser.add_argument('--monitor', action='store_true', help='')
parser.add_argument('--nonlin1st', action='store_true', help='place nonlinearity before normalization')
parser.add_argument('--testpth', action='store_true', help='')
parser.add_argument('--floodG', type=float, default=0, help='')
parser.add_argument('--floodD', type=float, default=0, help='')
parser.add_argument('--focal', action='store_true', help='')
parser.add_argument('--focal_alpha', type=float, default=0.25, help='')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='')
parser.add_argument('--fakepool', type=int, default=0, help='')
parser.add_argument('--layer_noise', type=float, default=0, help='')
parser.add_argument('--noiselayers', type=str, default="", help='')
parser.add_argument('--xlayers', type=int, default=0, help='')
parser.add_argument('--xDcorr', action='store_true', help='')


raw_args = " ".join(sys.argv)
print(raw_args)

opt = parser.parse_args()

if opt.monitor:
    import matplotlib.pyplot as plt
    from functools import partial

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id>=0:
        gpu_ids.append(id)

str_median = opt.medianindex.split(',')
median_ids = []
for mid in str_median:
    if (mid == ""): continue
    mid = int(mid)
    if mid>=0:
        median_ids.append(mid)
opt.median_ids = median_ids

str_dmedian = opt.dmedianindex.split(',')
dmedian_ids = []
for mid in str_dmedian:
    if (mid == ""): continue
    mid = int(mid)
    if mid>=0:
        dmedian_ids.append(mid)
opt.dmedian_ids = dmedian_ids

str_dropout = opt.dropoutindex.split(',')
dropout_ids = []
for mid in str_dropout:
    if (mid == ""): continue
    mid = int(mid)
    if mid>=0:
        dropout_ids.append(mid)
opt.dropout_ids = dropout_ids

str_attn = opt.attention.split(',')
attn_ids = []
for mid in str_attn:
    if (mid == ""): continue
    mid = int(mid)
    if mid>=0:
        attn_ids.append(mid)
opt.attn_ids = attn_ids

str_attnD = opt.attentionD.split(',')
attnD_ids = []
for mid in str_attnD:
    if (mid == ""): continue
    mid = int(mid)
    if mid>=0:
        attnD_ids.append(mid)
opt.attnD_ids = attnD_ids

str_noiseL = opt.noiselayers.split(',')
noise_ids = []
for mid in str_noiseL:
    if (mid == ""): continue
    mid = int(mid)
    if mid>=0:
        noise_ids.append(mid)
opt.noise_ids = noise_ids

if opt.save_everyD < 0:
    opt.save_everyD = opt.save_every

fG_ids = []
if opt.freezeG != "": 
  str_freezeG = opt.freezeG.split(',')
  for fgid in str_freezeG:
    fgid = int(fgid)
    if fgid>=0:
        fG_ids.append(fgid)

print(opt)

if opt.blur > 0:
    import kornia
    blurKernel = opt.blur
    if opt.blurtype == "box":            
        blur = kornia.filters.BoxBlur((blurKernel, blurKernel))
    elif opt.blurtype == "median":
        blur = kornia.filters.MedianBlur((blurKernel, blurKernel))
    else:
        assert(False, "Unknown blur type")
    #blur = kornia.filters.MedianBlur((blurKernel, blurKernel))

from nqmodel4 import _netG, _netD, _netE, weights_init, TVLoss

rundir = os.path.join(opt.runroot,opt.name)

try:
    os.makedirs(rundir)
except OSError:
    pass

try:
    os.makedirs(os.path.join(rundir, 'model'))
except OSError:
    pass

try:
    os.makedirs(os.path.join(rundir, 'visual'))
except OSError:
    pass

try:
    os.makedirs(os.path.join(rundir, 'frames'))
except OSError:
    pass


try:
    os.makedirs(os.path.join(rundir, 'results'))
except OSError:
    pass

try:
    os.makedirs(os.path.join(rundir, 'images'))
except OSError:
    pass

with open(os.path.join(rundir, "cmd.txt"), "w") as text_file:
    text_file.write(raw_args)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#opt.cuda=False
if torch.cuda.is_available():
    opt.cuda=True
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.cuda.set_device(gpu_ids[0])

    cudnn.benchmark = opt.benchmark
    cudnn.deterministic = opt.determ
    cudnn.enabled = opt.cudnn

if opt.dsmeta:
    import json
    with open(opt.dsmeta) as f:
        dsmeta = json.load(f)
        opt.dsmean = tuple(dsmeta['mean'])
        opt.dsstd = tuple(dsmeta['std'])
else:
    opt.dsmean = (0.5, 0.5, 0.5)
    opt.dsstd = (0.5, 0.5, 0.5)

if opt.cyclr > 0:
    opt.lrDmin = opt.lrD/opt.cyclr
    opt.lrGmin = opt.lrG/opt.cyclr
    opt.lrEmin = opt.lrE/opt.cyclr


transf = []
if opt.crop: # resize smaller side and then crop to center
    transf.append(transforms.Resize(opt.imageSize))
    transf.append(transforms.CenterCrop(opt.imageSize))
elif opt.randomcrop:
    transf.append(transforms.RandomCrop((opt.imageSize, opt.imageSize)))
else: # force into square
    transf.append(transforms.Resize((opt.imageSize, opt.imageSize)))

if opt.flip:
    transf.append(transforms.RandomHorizontalFlip())

if opt.brightness > 0 or opt.contrast > 0 or opt.saturation > 0:
    if opt.drandom:
        transf.append(transforms.ColorJitter(brightness=opt.brightness, contrast=opt.contrast, saturation=opt.saturation))
    else:
        br = (opt.brightness, opt.brightness) if opt.brightness > 0 else 0
        cr = (opt.contrast, opt.contrast) if opt.contrast > 0 else 0
        st = (opt.saturation, opt.saturation) if opt.saturation > 0 else 0
        transf.append(transforms.ColorJitter(brightness=br, contrast=cr, saturation=st))



transf.append(transforms.ToTensor())
transf.append(transforms.Normalize(opt.dsmean, opt.dsstd))

xform =transforms.Compose(transf)

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    dataset = dset.ImageFolder(root=opt.dataroot, transform=xform)              
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, drop_last=True, # !!!experiment
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = len(gpu_ids)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nef = int(opt.nef)
nc = 3
lmbd = opt.lmbd

assert(opt.imageSize in [64,128,256,512,1024, 2048])


def bw2rgb(im, out=True):
   im = torch.clamp(im, -1, 1)
   im = im.cpu().numpy()[0]
   if out:
      im = im/2 + 0.5
   #print(im.min(), im.max())
   im = (im*255).astype(np.uint8) #.transpose((1, 2, 0))
   #print(im.shape)
   bw = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
   im = bw.astype(np.float32)/255.
   im = torch.from_numpy(im.transpose(2,0,1))
   #print(im.shape) 
   return im

def rgb2bw(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# create models

# generator


if opt.instance:
    netG = _netG(ngpu, norm_layer=nn.InstanceNorm2d, opt=opt)
    if opt.init == "dcgan":
        netG.apply(weights_init)
elif opt.batchnorm:
    netG = _netG(ngpu, norm_layer=nn.BatchNorm2d, opt=opt)
    if opt.init == "dcgan":
        netG.apply(weights_init)
elif opt.nonorm:
    netG = _netG(ngpu, norm_layer=None, opt=opt)
    if opt.init == "dcgan":
        netG.apply(weights_init)
else:
    netG = _netG(ngpu, opt=opt)
    if opt.init == "dcgan":
        netG.apply(weights_init)

if opt.orthoG:
    for m in netG.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal(m.weight)

# TEST CODE
'''
def getLevel2G(l):
    level = -1 
    down = False
    if "deconv" in l:
        level = l.replace("deconv", "")
        #key = "deconv"+level
    elif "upblock" in l:
        level = l.replace("upblock", "")
        down = True # = "upblock"+level+".blocks.deconv"+level
    elif "in" in l:
        level = 0
        #key = l
    elif "out" in l:
        level = -1
        #key = l      
    return int(level), down
    

maxlevel = 0
G_groups = {}
for key, mod in netG.main.named_children():
    level, down = getLevel2G(key)
    if level > maxlevel:
        maxlevel = level
    print(key, level)
    
    if not down:  # add the layer directly
        if level in G_groups:
            G_groups[level].append(key)
        else:
            G_groups[level] = [key]
    else:  # add the children layers instead
        for m in mod.children():  # blocks, shortcut, activation
          for key, submod in m.named_children():
            #print(key, submod)  
            if level in G_groups:
                G_groups[level].append(key)
            else:
                G_groups[level] = [key]
            
                
G_groups[maxlevel+1] = G_groups[-1]    
del G_groups[-1]    
print(G_groups)            

#exit()

# ENDS

'''


# load prelearned weights if any

if opt.netG != '':
    Gpar = torch.load(opt.netG)
    Gpar_ = torch.load(opt.netG)
    Gpar = {}
    for k in Gpar_:
        k1 = k
        if "deconv" in k and not "upblock" in k:
             split = k.split('.')
             n = split[1].replace('deconv','')
             k1 = k.replace("deconv"+n, "upblock"+n+".blocks.deconv"+n)
             print(k1)
        elif "conv" in k and not "upblock" in k: # upsample layer 
             split = k.split('.')
             n = split[1].replace('conv','')
             k1 = k.replace("conv"+n, "upblock"+n+".blocks.conv"+n)
             print(k1)

        Gpar[k1] = Gpar_[k]
    try:
      netG.load_state_dict(Gpar, strict = not opt.nostrict)
    except RuntimeError:
      print("Layer size mismatch during loading")

if opt.cuda:
        netG.cuda() 

if 'G' in opt.dlr: 
        highlrG = []
        lowlrG = []
        newPar = netG.state_dict()
        for k in newPar.keys():
           if k not in Gpar:
               highlrG.append(k)
               #print(k, newPar[k].shape, "not in loaded file")
           else:
               lowlrG.append(k) 

        print("G using high LR for ", highlrG) 
        print("G using low LR for ", lowlrG) 
 
        hGmod = []
        lGmod = []
        for k, m in netG.named_modules():
            print(k)
            k_ = k + ".weight"
            if k_ in highlrG:
                hGmod.append(m) #.parameters())
            if k_ in lowlrG:
                lGmod.append(m) #.parameters())

        #print(hGmod)
        #print(lGmod)
        #print("------------------------")

        if opt.testpth:
          print("netG: comparing present model with loaded pth") 
          newPar = netG.state_dict()
          for k in newPar.keys():
            if k in Gpar:
                print(k, newPar[k].shape, Gpar[k].shape)
            else:             
                print(k, newPar[k].shape, "not in loaded file")
          for k in Gpar.keys():
            if k not in newPar.keys():
                print(k, Gpar[k].shape, "not in new model")


# discriminator

if opt.instance:
    netD = _netD(ngpu, use_sigmoid=(not opt.lsgan), norm_layer=nn.InstanceNorm2d, opt=opt)
    if opt.init == "dcgan":
        netD.apply(weights_init)
else:
    netD = _netD(ngpu, use_sigmoid=(not opt.lsgan), opt=opt)
    if opt.init == "dcgan":
        netD.apply(weights_init)
        
if opt.orthoD:
    for m in netD.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal(m.weight)        


# load prelearned weights if any

if opt.netD != '':
    Dpar = torch.load(opt.netD)

    if opt.xDcorr and opt.xlayers > 0:
        xltkeys = ['conv0d', 'conv0c','conv0b','conv0', 'norm0d', 'norm0c','norm0b','norm0']
        xlt2keys = ['conv0e','conv0d','conv0c', 'conv0b', 'norm0e','norm0d','norm0c', 'norm0b'] 
        #xltkeys = xltkeys[opt.xlayers]
        print(xltkeys)
        for p_ in Dpar.keys():
            for xl in xltkeys:
                if xl in p_.split('.'):
                    xl2 = xlt2keys[xltkeys.index(xl)]                
                    p2 = p_.replace(xl, xl2)
                    print(p_, p2)
                    Dpar[p2] = Dpar[p_]
                    #TODO delete the irrelevant key(s)
                    #del Dpar[p_]
        print(Dpar.keys()) 

    try:
      netD.load_state_dict(Dpar, strict = not opt.nostrict)
    except RuntimeError:
      print("Layer size mismatch during loading") 

if opt.cuda:
        netD.cuda()

if 'D' in opt.dlr: 
        highlrD = []
        lowlrD = []
        newPar = netD.state_dict()
        for k in newPar.keys():
           if k not in Dpar:
               highlrD.append(k)
               #print(k, newPar[k].shape, "not in loaded file")
           else:
               lowlrD.append(k) 

        print("D using high LR for ", highlrD) 
        print("D using low LR for ", lowlrD) 

        hDmod = []
        lDmod = []
        for k, m in netD.named_modules():
            k_ = k + ".weight"
            if k_ in highlrD:
                hDmod.append(m) #.parameters())
            if k_ in lowlrD:
                lDmod.append(m) #.parameters())

        #print(hDmod)
        #print(lDmod)
        #print("---------------------------")

        if opt.testpth:
          print("netD: comparing present model with loaded pth") 
          newPar = netD.state_dict()
          for k in newPar.keys():
            if k in Dpar:
                print(k, newPar[k].shape, Dpar[k].shape)
            else:             
                print(k, newPar[k].shape, "not in loaded file")
          for k in Dpar.keys():
            if k not in newPar.keys():
                print(k, Dpar[k].shape, "not in new model")


print(netD)

# encoder

if not opt.withoutE:
    if opt.instance:
        netE = _netE(ngpu, use_sigmoid=False, norm_layer=nn.InstanceNorm2d, opt=opt)
        if opt.init == "dcgan":
            netE.apply(weights_init)
    else:
        netE = _netE(ngpu, use_sigmoid=False, opt=opt)
        if opt.init == "dcgan":
            netE.apply(weights_init)

    # load prelearned weights if any

    if opt.netE != '':
        Epar = torch.load(opt.netE)
        try:
          netE.load_state_dict(Epar, strict = not opt.nostrict)   
        except RuntimeError:
          print("Layer size mismatch during loading") 


    print(netE)

def getLevel(str):
    level = -1 
    key = ""
    if "deconv" in str:
        level = str.replace("deconv", "")
        key = "deconv"+level
    elif "upblock" in str:
        level = str.replace("upblock", "")
        key = "upblock"+level+".blocks.deconv"+level
    return int(level), key

if len(fG_ids) > 0:
    #print(Gpar.keys())
    for key, mod in netG.main.named_children():
      level, newkey = getLevel(key)
      if level < 0:
          continue
      if level in fG_ids:
          #if k[0] == "outconv": continue #do not freeze the final output layer
          layer = "main."+newkey
          #print(level, layer)
          w = layer +".weight"
          if w in Gpar.keys():	     
              print("freezing netG."+w)
              mod.requires_grad = False

if not opt.flip_labels:
  real_label = opt.reallabel
  fake_label = opt.fakelabel
else:
  real_label = 0.0
  fake_label = 1.0
        

# loss module for real / fake testing

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=real_label, target_fake_label=fake_label, noisy=False, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.noisy = noisy
        if use_lsgan:
            self.loss = nn.MSELoss()
        elif opt.focal:
            self.loss = WeightedFocalLoss(opt.focal_alpha, opt.focal_gamma)
        #elif opt.ssim:
        #    self.loss = kornia.SSIM(5)
        else:
            self.loss = nn.BCELoss()

    # make a target tensor for real and fake
    # use noisy labels if opt.nlabels
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            if self.noisy:
              real_tensor = self.Tensor(input.size()).uniform_(0.8, 1.0)
            else:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
            self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            if self.noisy:
                fake_tensor = self.Tensor(input.size()).uniform_(0, 0.2)
            else:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
            self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# additional loss functions for AE loss

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()

criterion = GANLoss(use_lsgan=opt.lsgan, tensor=torch.cuda.FloatTensor)
Dcriterion = GANLoss(use_lsgan=opt.lsgan, noisy = opt.nlabels, tensor=torch.cuda.FloatTensor)
criterionL1 = nn.L1Loss()
tvloss = TVLoss(opt.tvloss)

# general purpose vectors

input_ = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)


print(opt.cuda)
if opt.cuda:
    #hGmod.cuda()
    #lGmod.cuda()
    #hDmod.cuda()
    #lDmod().cuda()
    #netD.cuda()
    #netG.cuda()
    if not opt.withoutE:
        netE.cuda()
    criterion.cuda()
    criterionL1.cuda()
    Dcriterion.cuda()
    input_, label = input_.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

def get_params(layers):
        params = []
        for layer in layers:
            #print("*",layer)
            if (hasattr(layer, "requires_grad") and layer.requires_grad):
                #To implement for custom nn.Parameter()
                print("Custom nn.Parameter() not supported")
            if(hasattr(layer, "weight") and hasattr(layer.weight, "requires_grad") and layer.weight.requires_grad):
                params.append(layer.weight)
            if (hasattr(layer, "bias") and hasattr(layer.bias, "requires_grad") and layer.bias.requires_grad):
                params.append(layer.bias)
        return params

def dlr_params(hmod, lmod, lr):
   #print("________>",get_params(hmod))
   #print("<_______",get_params(lmod))
   params = [{'params': get_params(hmod), 'lr': lr},
             {'params': get_params(lmod), 'lr': lr / opt.dlrrange}
            ]
   return params  
    
if 'G' in opt.dlr:
    paramsG = dlr_params(hGmod, lGmod, opt.lrG)
    lr_arrG = [opt.lrG, opt.lrG/opt.dlrrange]
else:
    paramsG = netG.parameters()

if 'D' in opt.dlr:
    paramsD = dlr_params(hDmod, lDmod, opt.lrD)
    lr_arrD = [opt.lrD, opt.lrD/opt.dlrrange]
else:
    paramsD = netD.parameters()

#if 'D' in opt.dlr:
#    print("D lr:", lr_arrD)
#if 'G' in opt.dlr:
#    print("G lr:", lr_arrG)

# setup optimizer
if opt.cyclr < 0:
    optimizerD = optim.Adam(paramsD, lr=opt.lrD, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    optimizerG = optim.Adam(paramsG, lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    if not opt.withoutE:
        optimizerE = optim.Adam(netE.parameters(), lr=opt.lrE, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
elif opt.dlr != '':
    if 'D' in opt.dlr:
        optimizerD = optim.RMSprop(paramsD, lr=opt.lrD, weight_decay=opt.weight_decay)
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD, weight_decay=opt.weight_decay)
    if 'G' in opt.dlr:
        #print("G lr:", lr_arrG)
        optimizerG = optim.RMSprop(paramsG, lr=opt.lrG, weight_decay=opt.weight_decay)
    #else:
    #    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG, weight_decay=opt.weight_decay)
    if not opt.withoutE:
        optimizerE = optim.RMSprop(netE.parameters(), lr=opt.lrE, weight_decay=opt.weight_decay)
else:    
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD, weight_decay=opt.weight_decay)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG, weight_decay=opt.weight_decay)
    if not opt.withoutE:
        optimizerE = optim.RMSprop(netE.parameters(), lr=opt.lrE, weight_decay=opt.weight_decay)


schedulers = []
if opt.cyclr < 0:
    schedulers.append(lr_scheduler.StepLR(optimizerD, step_size=opt.step, gamma=opt.gamma))
    schedulers.append(lr_scheduler.StepLR(optimizerG, step_size=opt.step, gamma=opt.gamma))
    if not opt.withoutE:
        schedulers.append(lr_scheduler.StepLR(optimizerE, step_size=opt.step, gamma=opt.gamma))
elif opt.dlr != '':
    if 'G' in opt.dlr: 
        schedulers.append(lr_scheduler.CyclicLR(optimizerG, base_lr=[lr / opt.cyclr for lr in lr_arrG], max_lr=lr_arrG, cycle_momentum=False, gamma=opt.gamma))
    else:
        schedulers.append(lr_scheduler.CyclicLR(optimizerG, opt.lrGmin, opt.lrG, cycle_momentum=False, gamma=opt.gamma))
    if 'D' in opt.dlr:        
        schedulers.append(lr_scheduler.CyclicLR(optimizerD, base_lr=[lr / opt.cyclr for lr in lr_arrD], max_lr=lr_arrD, cycle_momentum=False, gamma=opt.gamma))
    else:
        schedulers.append(lr_scheduler.CyclicLR(optimizerD, opt.lrDmin, opt.lrD, cycle_momentum=False, gamma=opt.gamma))
    if not opt.withoutE:
        schedulers.append(lr_scheduler.CyclicLR(optimizerE, opt.lrEmin, opt.lrE, cycle_momentum=False, gamma=opt.gamma))            
else:
    schedulers.append(lr_scheduler.CyclicLR(optimizerD, opt.lrDmin, opt.lrD, cycle_momentum=False, gamma=opt.gamma))
    schedulers.append(lr_scheduler.CyclicLR(optimizerG, opt.lrGmin, opt.lrG, cycle_momentum=False, gamma=opt.gamma))
    if not opt.withoutE:
        schedulers.append(lr_scheduler.CyclicLR(optimizerE, opt.lrEmin, opt.lrE, cycle_momentum=False, gamma=opt.gamma))


def clampWeightsD(m):
    if type(m) != nn.BatchNorm2d and type(m) != nn.Sequential:
      for p in m.parameters():
        p.data.clamp_(-opt.clipD, opt.clipD)

def clampWeightsG(m):
    if type(m) != nn.BatchNorm2d and type(m) != nn.Sequential:
      for p in m.parameters():
        p.data.clamp_(-opt.clipG, opt.clipG)

def clampWeightsE(m):
    if type(m) != nn.BatchNorm2d and type(m) != nn.Sequential:
      for p in m.parameters():
        p.data.clamp_(-opt.clipE, opt.clipE)



# autograd compatible resize  
def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data



# main training loop starts here

if opt.monitor:
    actG_means = [[] for _ in netG.modules()]
    actG_std = [[] for _ in netG.modules()]
    actD_means = [[] for _ in netD.modules()]
    actD_std = [[] for _ in netD.modules()]

    def append_stats_netG(i, mod, inp, outp):
        actG_means[i].append(outp.data.mean())
        actG_std[i].append(outp.data.std())

    def append_stats_netD(i, mod, inp, outp):
        actD_means[i].append(outp.data.mean())
        actD_std[i].append(outp.data.std())
    
    '''
    i = 0
    for _, m in enumerate(netG.modules()):
        classname = m.__class__.__name__
        if "UpBlock" in classname:
            print(classname) 
            m.register_forward_hook(partial(append_stats_netG, i))
            i += 1

    i = 0
    for _, (name, m) in enumerate(netD.named_modules()):
        
        #print(name)
        if ("conv" in name) and not "attn" in name:
            print(name) 
            m.register_forward_hook(partial(append_stats_netD, i))
            i += 1
    '''

if opt.fakepool > 0:
    fakepool = ImagePool(opt.fakepool)        
    

imgCtr = 0
blurCtr = opt.blurstep
blurState = True
for epoch in range(opt.niter):
    saveCtr = 0
    # get a batch of input images
    for i, data in enumerate(dataloader, 0):
        if opt.noise > 0:  # TUTKI TÄMÄ !!!!
            addnoise = torch.FloatTensor(data[0].size()).normal_(0, opt.noise)
            data[0] += addnoise
        iimgs = data[0].clone() # store original input images for later display

        # convert to bw if needed
        if opt.nc == 1:
          data_ = torch.Tensor(data[0].size(0), opt.nc, data[0].size(2), data[0].size(3))
          n = 0
          for im in data[0]:
            data_[n] = rgb2bw(im)
            iimgs[n] = bw2rgb(data_[n], out=True)
            n = n + 1
          real_cpu = data_
        else:
          real_cpu, _ = data    

        #if opt.brightness != 1:
        #    real_cpu = transforms.functional.adjust_brightness(real_cpu, opt.brightness)
        #if opt.contrast != 1:
        #    real_cpu = transforms.functional.adjust_contrast(real_cpu, opt.contrast)
        #if opt.dgamma != 1:
        #    real_cpu = transforms.functional.adjust_gamma(real_cpu, opt.dgamma)


        #if blurKernel > 0:            
        #    blur = kornia.filters.BoxBlur((blurKernel, blurKernel))
        #    blurred = torch.zeros_like(real_cpu)
        #    n = 0
        #    for im in real_cpu:
        #        #print(n, blurKernel)
        #        blurred[n] = blur(im.unsqueeze(0)).squeeze()
        #        n = n + 1
        #    real_cpu = blurred


        # update netD 

        # first with real images
        netD.zero_grad()
        batch_size = real_cpu.size(0) # needed for take care of an incomplete batch at the end of an epoch 
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        inputv = input_.resize_as_(real_cpu).copy_(real_cpu)
        
        if opt.blur > 0 and blurKernel > 0 and blurState:
            inputv = blur(inputv.detach())            
            
            if opt.blurnoise > 0:
                blurnoise = torch.FloatTensor(data[0].size()).normal_(0, opt.blurnoise)
                if opt.cuda:
                    blurnoise = blurnoise.cuda()
                inputv = inputv + blurnoise

        input_samples = inputv.clone()
        outputr = netD(inputv.detach())                 # get D(x) for real images
        errD_real = Dcriterion(outputr, True)  # get err ref to real
        errD_real.backward()                   # get D_real gradients

        D_x = outputr.data.mean()

        # now D with fake images
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1) # make random z
        noisev = noise.detach() 
        fake_z = netG(noisev.detach()) 		          # G(z), why detach?

        if opt.fakepool > 0:
            fake_zp = fakepool.query(fake_z.detach())
            #fake_zp = fakepool.query(fake_z.clone().detach()) # TRY NEXT???
            output = netD(fake_zp.detach())                    # D(G(z))
        else:
            output = netD(fake_z.detach())                    # D(G(z))
        errD_fake = Dcriterion(output, False)             # get err ref to fake

        errD_fake.backward()                              # get D_fake gradients
        D_G_z1 = output.data.mean()

        errD = errD_real + errD_fake                      # total D err display
        
        if opt.floodD > 0:
            errD = abs(errD - opt.floodD) + opt.floodD    
        
        
        olD = 0
        if opt.orthoD:
            #with torch.enable_grad():
            reg = 1e-6 #.cuda()
            orth_loss = torch.zeros(1).cuda()
            for name, param in netD.named_parameters():
                #print('netD', name, param.shape)
                if 'bias' not in name:
                    param = param.cuda()
                    param_flat = param.view(param.shape[0], -1) #.cuda()
                    #if param_flat.shape[0] < 2:
                    #    continue
                    #print(param_flat)
                    sym = torch.mm(param_flat, torch.t(param_flat)) #cuda()
                    #print(sym)
                    #print(param_flat.shape[0])
                    symd = torch.eye(param_flat.shape[0]).cuda()
                    #print(symd)
                    sym -= symd
                    orth_loss = orth_loss + (reg * sym.abs().sum())
            orth_loss.backward()
            olD = orth_loss.cpu().detach().numpy()

        l1reg = None
        l1D = 0
        if opt.l1reg > 0:
            for name, param in netD.named_parameters():
                if l1reg is None:
                    l1reg = param.norm(1)
                else:
                    l1reg = l1reg + param.norm(2)
            l1reg = opt.l1reg * l1reg
            l1reg.backward()
            l1D = l1reg.cpu().detach().numpy()
            
        optimizerD.step()                                 # update D weights
      
        if opt.clipD > 0:
            netD.apply(clampWeightsD)
            #netD.zero_grad() # Doing this here will prevent generator from learning !!! Why?

        #
        # Update netG 
        # needs:
        # netD, updated above
        # fake_Z, created by netG above from noisev
        # inputv, batch of real images from dataset
        # output D(G(fake_z))
        # outputr D(x) x is real images from dataset
        netG.zero_grad()

        output = netD(fake_z)  # evaluate fake_z again with D after optim, comment to use eval with earlier netD
        tvl = 0

        # if encoder used
        if not opt.withoutE and lmbd > 0:
            # encode real images into z and back, test result with D 
            embedding = netE(inputv.detach()).view(batch_size,opt.nz,1,1)  # use E to get z = E(x) for real images 
            fake_e = netG(embedding.detach())                              # new fake = G(E(x)), detach from E, train E later
            dist = l1_loss(inputv.detach(), fake_e)*lmbd                   # get err between input and G(E(x))
    
            errGadv = criterion(output, True)                                 # get D(G(fake_z)) err
            if opt.rsgan:
                errGadv = errGadv - criterion(outputr.detach(), True)            # decrease prob that real seen as real !!!  

            errG = errGadv + dist				  # add adversarial and AE loss
            if opt.tvloss: 
                tvl = tvloss(fake_z)                              # tv loss on G(E(x))
                errG = errG + tvl
        else:
            errG = criterion(output, True)	# no E used, take plain gan loss as errG
            if opt.rsgan:
                # use outputr on next line instead of using netD again??? 
                errG = errG - criterion(netD(inputv.detach()), True)       # decrease prob that real seen as real !!!  
                #errG = errG - criterion(outputr.detach(), True)             # decrease prob that real seen as real !!!  
            if opt.tvloss: 
                tvl = tvloss(fake_z)            # just add TV loss on G(z)
                errG = errG + tvl
            dist = 0
        
        if opt.floodG > 0:
            errG = abs(errG - opt.floodG) + opt.floodG    

        olG = 0
        if opt.orthoG:
                #with torch.enable_grad():
                reg = 1e-6
                orth_loss = torch.zeros(1).cuda()
                for name, param in netG.named_parameters():
                    #print('netG', name, param.shape)
                    if 'bias' not in name:
                        param = param.cuda()
                        param_flat = param.view(param.shape[0], -1).cuda()
                        #print(param_flat)
                        #if param_flat.shape[0] < 2:
                        #    continue
                        sym = torch.mm(param_flat, torch.t(param_flat))
                        symd = torch.eye(param_flat.shape[0]).cuda()
                        #print(sym, symd)
                        sym = sym - symd
                        orth_loss = orth_loss + (reg * sym.abs().sum())
                    orth_loss.backward(retain_graph = True)
                    olG = orth_loss.cpu().detach().numpy()
                #errG = errG + orth_loss 

        l1reg = None
        l1G = 0
        if opt.l1reg > 0:
            for name, param in netG.named_parameters():
                if l1reg is None:
                    l1reg = param.norm(1)
                else:
                    l1reg = l1reg + param.norm(2)
            l1reg = opt.l1reg * l1reg
            l1reg.backward()
            l1G = l1reg.cpu().detach().numpy()
            #errG = errG + opt.l1reg * l1reg

        errG.backward(retain_graph = (not opt.withoutE))   # get G gradients, need to retain graph if encoder is used
        D_G_z2 = output.data.mean() 
        ### TODO with fakepool D_G_z1 is now obtained with a different set (only affects printouts?)
        # DGz1 is taken before D update and DGz2 after
        optimizerG.step()                        # update G parameters

        if opt.clipG > 0:
            netG.apply(clampWeightsG)
            netG.zero_grad() # needed here or not?


        # Update E
        if not opt.withoutE:
            netE.zero_grad()
            netG.zero_grad() #just to be sure 
            #embedding = netE(fake_z.detach())  # E(G(z))
            if opt.trainE == "image": # optimize image => E => z => G => image2
                embedding = netE(inputv.detach()).view(batch_size,opt.nz,1,1)  # z = E(x)
                embZstat = (embedding.data.mean().cpu().numpy(), embedding.data.std().cpu().numpy()) 
                fake_e = netG(embedding)  
                errE = mse_loss(fake_e, inputv.detach()) # err between E(G(z)) and z
            else: # optimize Z => G => image => encoder => Z2
                # fake_z = netG(noisev.detach())
                z_enc = netE(fake_z) #.detach())
                embZstat = (z_enc.data.mean().cpu().numpy(), z_enc.data.std().cpu().numpy()) 
                errE = l1_loss(z_enc.view(batch_size, opt.nz, 1, 1), noisev.detach()) # err between E(G(z)) and z
                #errE = l1_loss(noisev.detach(), z_enc)                                
            errE.backward()
            optimizerE.step()
            #optimizerG.step()  # should we really update G here too or not?

            if opt.clipE > 0:
                netE.apply(clampWeightsE)
                netE.zero_grad() # needed here or not?
            

            print('[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f / %.2f  Loss_E: %.2f D(x): %.6f D(G(z)): %.6f / %.6f Dist: %.2f TVLoss: %.2f Ez stat %.2f / %.2f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data, errG.data, errGadv.data, errE.data, D_x, D_G_z1, D_G_z2,  dist, tvl, embZstat[0], embZstat[1]))
        else:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f  D(x): %.8f D(G(z)): %.8f / %.8f L1 D/G: %4f, %4f'
                    % (epoch, opt.niter, i, len(dataloader),
                    errD.data, errG.data, D_x, D_G_z1, D_G_z2, l1D, l1G))
        
 
        # save single samples if opt.imgStep > 0 
        if opt.imgStep != 0 and imgCtr % opt.imgStep == 0:
            if opt.fixednoise:
                sampleNoise = fixed_noise.clone() 
            else:
                sampleNoise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
            if opt.cuda: sampleNoise = sampleNoise.cuda()
            fakeimg = netG(Variable(sampleNoise))
            fakeimg = fakeimg.data
            for fidx in range(0, opt.batchSize):
                singleFake = fakeimg[fidx]
                vutils.save_image(singleFake,
                    opt.runroot+'/'+opt.name+'/images/sample%06d-%d.png' % (int(imgCtr/opt.imgStep), fidx),
                    normalize=True)        

        imgCtr = imgCtr + 1

        # visualize results
        if i % 100 == 0:
 
            vutils.save_image(iimgs,
                    opt.runroot+'/%s/visual/real_samples.png' % opt.name,
                    normalize=True)
            vutils.save_image(input_samples,
                    opt.runroot+'/%s/visual/input_samples.png' % opt.name,
                    normalize=True)
            fake = netG(fixed_noise)
            fake = fake.data
            if opt.nc == 1:
               fake_ = torch.Tensor(fake.size(0), 3, fake.size(2), fake.size(3))
               #print(fake_.shape)
               n = 0
               for im in fake:
                  fake_[n] = bw2rgb(im)
                  n = n + 1
               fake = fake_

            #print('saving fakes ', fake.min(), fake.mean(), fake.std(), fake.max())
            vutils.save_image(fake,
                    opt.runroot+'/%s/visual/fake_samples_epoch_%03d.png' % (opt.name, epoch),
                    normalize=True)
            if opt.saveGextra:
                torch.save(netG.state_dict(), opt.runroot+'/%s/model/netG_epoch_%d_%d.pth' % (opt.name, epoch, saveCtr))
                saveCtr = saveCtr + 1
        
        if opt.cyclr > 0:
            for scheduler in  schedulers:
                scheduler.step()


    # do checkpointing
    if epoch % opt.save_every == 0:
        torch.save(netG.state_dict(), opt.runroot+'/%s/model/netG_epoch_%d.pth' % (opt.name, epoch))
        #torch.save(netD.state_dict(), './runs/%s/model/netD_epoch_%d.pth' % (opt.name, epoch))
        if not opt.withoutE:
            torch.save(netE.state_dict(), opt.runroot+'/%s/model/netE_epoch_%d.pth' % (opt.name, epoch))

    if epoch % opt.save_everyD == 0:
        #torch.save(netG.state_dict(), opt.runroot+'/%s/model/netG_epoch_%d.pth' % (opt.name, epoch))
        torch.save(netD.state_dict(), opt.runroot+'/%s/model/netD_epoch_%d.pth' % (opt.name, epoch))

    if opt.monitor:
        i = 0
        for _, m in enumerate(netG.modules()):
            classname = m.__class__.__name__
            if "Attn" in classname:
                print(classname)
                attn_map = m.map #torch.sum(m.map, 1).unsqueeze(1).repeat(1,3,1,1)
                cmax = attn_map.shape[1]
                print(attn_map.shape)
                for c in range(0, cmax):
                    cmap = attn_map[:,c,:,:].unsqueeze(1).repeat(1,3,1,1)
                    print(c, cmap.shape)
                    vutils.save_image(cmap,
                        opt.runroot+'/%s/debug/netG_attention_epoch_%03d_%03d_%03d.png' % (opt.name, epoch, i, c),
                        normalize=True) 
                i += 1

        for _, m in enumerate(netD.modules()):
            classname = m.__class__.__name__
            if "Attn" in classname:
                print(classname)
                attn_map = m.map #torch.sum(m.map, 1).unsqueeze(1).repeat(1,3,1,1)
                cmax = attn_map.shape[1]
                print(attn_map.shape)
                for c in range(0, cmax):
                    cmap = attn_map[:,c,:,:].unsqueeze(1).repeat(1,3,1,1)
                    print(c, cmap.shape)
                    vutils.save_image(cmap,
                        opt.runroot+'/%s/debug/netD_attention_epoch_%03d_%03d_%03d.png' % (opt.name, epoch, i, c),
                        normalize=True) 
                i += 1
        '''
        for o in actD_means: plt.plot(o)
        plt.legend(range(6))
        #plt.show()
        plt.savefig("netD.png")
        plt.clf()

        for o in actD_std: plt.plot(o)
        plt.legend(range(6))
        #plt.show()
        plt.savefig("netDstd.png")
        plt.clf()

        for o in actG_std: plt.plot(o)
        plt.legend(range(6))
        #plt.show()
        plt.savefig("netGstd.png")
        plt.clf()

        for o in actG_means: plt.plot(o)
        plt.legend(range(6))
        #plt.show()
        plt.savefig("netG.png")
        plt.clf()
        '''

    #step lrRate
    if opt.cyclr <= 0:
        for scheduler in  schedulers:
            scheduler.step()

    if opt.blur > 0 and opt.blurmode == "alt":
        blurState = not blurState
        print("blur set to "+str(blurState))

    blurCtr = blurCtr - 1
    if (opt.blur > 0) and (blurCtr <= 0) and (blurKernel > 0):
        blurCtr = opt.blurstep
        blurKernel = blurKernel - opt.blurdelta 
        if blurKernel <= 0:
            blurKernel = 0    
