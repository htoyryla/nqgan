import argparse
import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from nqmodel4 import _netG
from random import random

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline', type=str, help='trained model name')
parser.add_argument('--which_epoch', default='24', type=str, help='0,1,2,3,4...')
parser.add_argument('--batchSize', default=32, type=int, help='batchsize')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--imageSize', default=256, type=int, help='image size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--nolin', action='store_true', help='use some linear layers in models')
parser.add_argument('--instance', action='store_true', help='use instance norm')
parser.add_argument('--batchnorm', action='store_true', help='use batch norm')
parser.add_argument('--nonorm', action='store_true', help='no norm layer')
parser.add_argument('--upsample', action='store_true', help='use upsample and conv instead of convtranspose')
parser.add_argument('--howMany', default=100, type=int, help='how many transitions')
parser.add_argument('--steps', default=20, type=int, help='interpolation steps')
parser.add_argument('--cols', default=4, type=int, help='interpolation steps')
parser.add_argument('--median', type=int, default=0, help='median layer kernel, set 0 for no median filter')
parser.add_argument('--medianindex', type=str, default="", help='')
parser.add_argument('--medianout', type=int, default=0, help='')
parser.add_argument('--output', default='frames', type=str, help='output directory')
parser.add_argument('--x90', action='store_true', help='')
parser.add_argument('--x270', action='store_true', help='')
parser.add_argument('--rhflip', action='store_true', help='')

opt = parser.parse_args()
opt.debug = False
#opt.nolin = True
print(opt)

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

str_median = opt.medianindex.split(',')
median_ids = []
for mid in str_median:
    if (mid == ""): continue
    mid = int(mid)
    if mid>=0:
        median_ids.append(mid)
opt.median_ids = median_ids

modeldir = outdir = os.path.join('./runs',opt.name,'model')
outdir = os.path.join('./runs',opt.name, opt.output)
print(outdir)

try:
    os.makedirs(outdir)
except OSError:
    pass

#---------------generate images
def generate_img(model):
    ctr = 0
    start_noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).normal_(0,1)
    start_noise = start_noise.cuda()
    start_noise = Variable(start_noise)
    s = opt.imageSize
    for i in range(opt.howMany):
        end_noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).normal_(0,1)
        end_noise = end_noise.cuda()
        end_noise = Variable(end_noise)
        delta = (end_noise - start_noise)/opt.steps 
        for j in range (opt.steps):
          outputs = model(start_noise + j * delta)
          fake = outputs.data
          print(ctr)
          w = opt.cols * s
          rows = int(opt.batchSize/opt.cols)
          h = rows * s
          print(h,w)
          oimg = torch.ones(1, 3, h, w)
          #print(oimg.shape)
          for n in range(0,opt.batchSize):
              im = fake[n,:,:,:]
              row = n // opt.cols
              col = n % opt.cols
              x = col*s
              y = row*s
              #print(rows, row, opt.cols, col, x, y, im.shape)
              #print(oimg[0,:,x:x+s,y:y+s].shape)
              if opt.x270:
                  im = im.transpose(1,2).flip(1)
              elif opt.x90:
                  im = im.transpose(1,2)
              if opt.rhflip and random() > 0.5:
                  im = im.flip(2)
              oimg[0,:,y:y+s, x:x+s] = im
          torchvision.utils.save_image(
                    oimg, #.view(1, im.size(0), im.size(1), im.size(2)),
                    os.path.join(outdir, 'frame-%d_%d.png'%(int(opt.which_epoch),ctr)),
                    nrow=1,
                    padding=0,
                    normalize=True)
          ctr = ctr + 1            
        start_noise = end_noise        

#-----------------Load model
def load_network(network):
    save_path = os.path.join(modeldir,'netG_epoch_%s.pth'%opt.which_epoch)
    state_dict = torch.load(save_path)
    print(state_dict.keys())
    network.load_state_dict(state_dict)
    return network

if opt.instance:
    netG = _netG(norm_layer=nn.InstanceNorm2d, opt=opt)
elif opt.batchnorm:
    netG = _netG(norm_layer=nn.BatchNorm2d, opt=opt)
elif opt.nonorm:
    netG = _netG(ngpu, norm_layer=None, opt=opt)
else:
    netG = _netG(opt=opt)
print(netG)
model = load_network(netG)
model = model.cuda()

generate_img(model)
