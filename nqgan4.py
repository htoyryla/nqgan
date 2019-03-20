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
import torch.nn.functional as F

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
parser.add_argument('--step', type=int, default=40, help='lr step, default=40')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma, default=0.1')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--name', default='baseline', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0 0,1,2 0,2')
parser.add_argument('--lsgan', action='store_true', help='use lsgan')
parser.add_argument('--rsgan', action='store_true', help='use rsgan')
parser.add_argument('--D2', action='store_true', help='use additional convlayers in D')
parser.add_argument('--instance', action='store_true', help='use instance norm')
parser.add_argument('--batchnorm', action='store_true', help='use batch norm')
parser.add_argument('--nonorm', action='store_true', help='no norm layer')
parser.add_argument('--upsample', action='store_true', help='use upsample and conv instead of convtranspose')
parser.add_argument('--cudnn', action='store_true', help='use cudnn')
parser.add_argument('--withoutE', action='store_true', help='do not use Encoder Network')
parser.add_argument('--debug', action='store_true', help='show debug info')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight. Greatly helps convergence but leads to artifacts in images, not recommended.')
parser.add_argument('--nlabels', action='store_true', help='use noisy labels')
parser.add_argument('--nostrict', action='store_true', help='allow partial loading of pretrained nets')
parser.add_argument('--noise', type=float, default=0.0, help='input noise, default=0.')
parser.add_argument('--flip', action='store_true', help='random horizontal flip')
parser.add_argument('--crop', action='store_true', help='crop into square')
parser.add_argument('--trainE', default="Z", help='how encoder is used')
parser.add_argument('--init', default="default", help='default | dcgan')
parser.add_argument('--nolin', action='store_true', help='use some linear layers in models')
parser.add_argument('--clipD', type=float, default=0, help='clip discriminator weights to given value')
parser.add_argument('--clipG', type=float, default=0, help='clip generator weights to given value')
parser.add_argument('--Dsize', type=float, default=0, help='image size for discriminator for special effects')
parser.add_argument('--flip_labels', action='store_true', help='flip real and fake labels')
parser.add_argument('--median', type=int, default=0, help='median layer kernel, set 0 for no median filter')
parser.add_argument('--medianindex', type=str, default="", help='')
parser.add_argument('--medianout', type=int, default=0, help='')
parser.add_argument('--minibatchstd', action='store_true', help='use minibatch std in D')
parser.add_argument('--saveGextra', action='store_true', help='save G always when saving sample images')


raw_args = " ".join(sys.argv)
print(raw_args)

opt = parser.parse_args()

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

if opt.save_everyD < 0:
    opt.save_everyD = opt.save_every

print(opt)

from nqmodel4 import _netG, _netD, _netE, weights_init, TVLoss

rundir = os.path.join(opt.runroot,opt.name)

try:
    os.makedirs(rundir)
    os.makedirs(os.path.join(rundir, 'model'))
    os.makedirs(os.path.join(rundir, 'visual'))
    os.makedirs(os.path.join(rundir, 'frames'))
    os.makedirs(os.path.join(rundir, 'results'))
except OSError:
    pass

with open(os.path.join(rundir, "cmd.txt"), "w") as text_file:
    text_file.write(raw_args)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.cuda=False
if torch.cuda.is_available():
    opt.cuda=True
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.cuda.set_device(gpu_ids[0])

    cudnn.benchmark = False #opt.cudnn #True
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

transf = []
if opt.crop: # resize smaller side and then crop to center
    transf.append(transforms.Resize(opt.imageSize))
    transf.append(transforms.CenterCrop(opt.imageSize))
else: # force into square
    transf.append(transforms.Resize((opt.imageSize, opt.imageSize)))

if opt.flip:
    transf.append(transforms.RandomHorizontalFlip())

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

assert(opt.imageSize in [64,128,256,512,1024])


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

# load prelearned weights if any

if opt.netG != '':
    Gpar = torch.load(opt.netG)
    try:
      netG.load_state_dict(Gpar, strict = not opt.nostrict)
    except RuntimeError:
      print("Layer size mismatch during loading") 

print(netG)

# discriminator

if opt.instance:
    netD = _netD(ngpu, use_sigmoid=(not opt.lsgan), norm_layer=nn.InstanceNorm2d, opt=opt)
    if opt.init == "dcgan":
        netD.apply(weights_init)
else:
    netD = _netD(ngpu, use_sigmoid=(not opt.lsgan), opt=opt)
    if opt.init == "dcgan":
        netD.apply(weights_init)

# load prelearned weights if any

if opt.netD != '':
    Dpar = torch.load(opt.netD)
    try:
      netD.load_state_dict(Dpar, strict = not opt.nostrict)
    except RuntimeError:
      print("Layer size mismatch during loading") 


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

if not opt.flip_labels:
  real_label = 1.0
  fake_label = 0.0
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

if opt.cuda:
    netD.cuda()
    netG.cuda()
    if not opt.withoutE:
        netE.cuda()
    criterion.cuda()
    criterionL1.cuda()
    input_, label = input_.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
if not opt.withoutE:
    optimizerE = optim.Adam(netE.parameters(), lr=opt.lrE, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)


schedulers = []
schedulers.append(lr_scheduler.StepLR(optimizerD, step_size=opt.step, gamma=opt.gamma))
schedulers.append(lr_scheduler.StepLR(optimizerG, step_size=opt.step, gamma=opt.gamma))
if not opt.withoutE:
    schedulers.append(lr_scheduler.StepLR(optimizerE, step_size=opt.step, gamma=opt.gamma))

def clampWeights(m):
    if type(m) != nn.BatchNorm2d and type(m) != nn.Sequential:
      for p in m.parameters():
        p.data.clamp_(-opt.clipD, opt.clipD)

# autograd compatible resize  
def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data


# main training loop starts here



imgCtr = 0
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

        # update netD 

        # first with real images
        netD.zero_grad()
        batch_size = real_cpu.size(0) # needed for take care of an incomplete batch at the end of an epoch 
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        inputv = input_.resize_as_(real_cpu).copy_(real_cpu)
        
        outputr = netD(inputv.detach())                 # get D(x) for real images
        errD_real = Dcriterion(outputr, True)  # get err ref to real
        errD_real.backward()                   # get D_real gradients

        D_x = outputr.data.mean()

        # now D with fake images
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1) # make random z
        noisev = noise.detach() 
        fake_z = netG(noisev.detach()) 		          # G(z), why detach?

        output = netD(fake_z.detach())                    # D(G(z))
        errD_fake = Dcriterion(output, False)             # get err ref to fake

        errD_fake.backward()                              # get D_fake gradients
        D_G_z1 = output.data.mean()

        errD = errD_real + errD_fake                      # total D err display
        optimizerD.step()                                 # update D weights
      
        if opt.clipD > 0:
            netD.apply(clampWeights)
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

        errG.backward(retain_graph = (not opt.withoutE))   # get G gradients, need to retain graph if encoder is used
        D_G_z2 = output.data.mean() 
        # DGz1 is taken before D update and DGz2 after
        optimizerG.step()                        # update G parameters



        if opt.clipG > 0:
            netG.apply(clampWeights)
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
            optimizerG.step()  # should we really update G here too or not?
            

            print('[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f / %.2f  Loss_E: %.2f D(x): %.2f D(G(z)): %.2f / %.2f Dist: %.2f TVLoss: %.2f Ez stat %.2f / %.2f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data, errG.data, errGadv.data, errE.data, D_x, D_G_z1, D_G_z2,  dist, tvl, embZstat[0], embZstat[1]))
        else:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f  D(x): %.4f D(G(z)): %.4f / %.4f TVLss: %4f'
                    % (epoch, opt.niter, i, len(dataloader),
                    errD.data, errG.data, D_x, D_G_z1, D_G_z2, tvl))
        
 
        # save single samples if opt.imgStep > 0 
        if opt.imgStep != 0 and imgCtr % opt.imgStep == 0:
            sampleNoise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
            if opt.cuda: sampleNoise = sampleNoise.cuda()
            fakeimg = netG(Variable(sampleNoise))
            fakeimg = fakeimg.data
            vutils.save_image(fakeimg,
                './runs/'+opt.name+'images/sample%06d.png' % (int(imgCtr/opt.imgStep)),
                normalize=True)        

        imgCtr = imgCtr + 1

        # visualize results
        if i % 100 == 0:
 
            vutils.save_image(iimgs,
                    './runs/%s/visual/real_samples.png' % opt.name,
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
                    './runs/%s/visual/fake_samples_epoch_%03d.png' % (opt.name, epoch),
                    normalize=True)
            if opt.saveGextra:
                torch.save(netG.state_dict(), './runs/%s/model/netG_epoch_%d_%d.pth' % (opt.name, epoch, saveCtr))
                saveCtr = saveCtr + 1

    # do checkpointing
    if epoch % opt.save_every == 0:
        torch.save(netG.state_dict(), './runs/%s/model/netG_epoch_%d.pth' % (opt.name, epoch))
        #torch.save(netD.state_dict(), './runs/%s/model/netD_epoch_%d.pth' % (opt.name, epoch))
        if not opt.withoutE:
            torch.save(netE.state_dict(), './runs/%s/model/netE_epoch_%d.pth' % (opt.name, epoch))

    if epoch % opt.save_everyD == 0:
        #torch.save(netG.state_dict(), './runs/%s/model/netG_epoch_%d.pth' % (opt.name, epoch))
        torch.save(netD.state_dict(), './runs/%s/model/netD_epoch_%d.pth' % (opt.name, epoch))


    #step lrRate
    for scheduler in  schedulers:
        scheduler.step()
