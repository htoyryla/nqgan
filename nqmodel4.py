from __future__ import print_function
import argparse
import os
import random
import math
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
import functools
from collections import OrderedDict
from medianpool import MedianPool2d
from Attention import Self_Attn
from sinerelu import SineReLU

#
# code for models for traindcg2
# htoyryla 30 Jul 2018
#
# experimental
#

# earlier versions
# v.2c3 try naming of layers
# v.2c4 larger kernels on larger layers
# v.2c5 models changed so that in D & E, layers are added to input size as image size grows 
# v.2c6 added convlayers to discriminator to get down to 1x1
#       corrected 25.7.2018 handling of nc, nz, ngf, ndf without globals, added nef

# new version as of 30 Jul will not produce models extensible to larger image size
# model3b adds refactored versions of model created in a loop

nc = 3

def weights_init(m):
    classname = m.__class__.__name__
    #print(m)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    #elif classname.find('Linear') != -1:
    #    m.weight.data.normal_(0.1, 0.02)
    #    if hasattr(m.bias, 'data'):
    #        m.bias.data.fill_(0)
    elif classname.find('Batchnorm') !=-1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Zview(nn.Module):
	def __init__(self):
		super(Zview, self).__init__()
	def forward(self, x):
                out =  x.view(x.size()[0], -1, 2, 2)
                return out

class ZEview(nn.Module):
	def __init__(self):
		super(ZEview, self).__init__()
	def forward(self, x):
                out =  x.view(x.size()[0], -1)
                return out

class BatchStdConcat(nn.Module):
    """
    Add std to last layer group of disc to improve variance
    """
    def __init__(self, groupSize=4):
        super().__init__()
        self.groupSize=groupSize

    def forward(self, x):
        shape = list(x.size())                                              # NCHW - Initial size
        xStd = x.clone().view(self.groupSize, -1, shape[1], shape[2], shape[3])     # GMCHW - split batch as groups of 4  
        xStd -= torch.mean(xStd, dim=0, keepdim=True)                       # GMCHW - Subract mean of shape 1MCHW
        xStd = torch.mean(xStd ** 2, dim=0, keepdim=False)                  # MCHW - Take mean of squares
        xStd = (xStd + 1e-08) ** 0.5                                        # MCHW - Take std 
        xStd = torch.mean(xStd.view(int(shape[0]/self.groupSize), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
                                                                            # M111 - Take mean across CHW
        xStd = xStd.repeat(self.groupSize, 1, shape[2], shape[3])           # N1HW - Expand to same shape as x with one channel 
        return torch.cat([x, xStd], 1)
    
    def __repr__(self):
        return self.__class__.__name__ + '(Group Size = %s)' % (self.groupSize)

class PixelNormalization(nn.Module):
    """
    This is the per pixel normalization layer. This will devide each x, y by channel root mean square
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim # dummy for interchangeablity with other norm layers
    
    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + 1e-8) ** 0.5

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


# total variation loss module
# use as a loss module, not a layer

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# generator, duplicates my mygan architecture as of 4/2017  

class _netG(nn.Module):
    def __init__(self, ngpu=1, norm_layer=PixelNormalization, opt=None):
        super(_netG, self).__init__()
        assert(opt is not None)
        self.nz = opt.nz
        self.nc = opt.nc
        self.ngf = opt.ngf
        self.size = opt.imageSize
        self.ngpu = ngpu
        self.nolin = opt.nolin 
        self.noise_ids = opt.noise_ids
        #self.activation = nn.ReLU(inplace=True)
        #if opt.elu:
        #    self.activation = nn.ELU(inplace=True)

        #if norm_layer == nn.BatchNorm2d:
        #    norm_layer = nn.BatchNorm2d(eps = opt.bneps)

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        if opt.elu:
            self.activation = nn.ELU(inplace=True)
        elif opt.sinerelu:
            self.activation = SineReLU()    

        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d

        if opt.x2:
            self.size = int(self.size/2) 
        elif opt.x4:
            self.size = int(self.size/4) 

        if opt.nolin:
            upsample_steps = int(round(math.log(self.size, 2))) - 1
        else:
            upsample_steps = int(round(math.log(self.size, 2))) - 1  # Corrected 17.12.2018

        n512 = upsample_steps - 5

        def upsample_level(index, multin, multout, normL, bias, monitor):
 
            if opt.upsample:  # whole network built of upsample+conv layers
              layers = [ ("upsample"+str(index), nn.Upsample(scale_factor=2, mode='nearest')),
                           ("replpad"+str(index), nn.ReplicationPad2d(1)), 
                           ("conv"+str(index), nn.Conv2d(self.ngf*multin, self.ngf*multout, 3, 1, 0, bias=bias))]

            # note: opt.upsample_after default = 1000 so this takes care also for deconv only networks
            elif opt.upsample_after >= 0 and index < opt.upsample_after :  # using deconv for lower and upsample+conv for higher
              if index < upsample_steps :
                layers = [("deconv"+str(index), nn.ConvTranspose2d(self.ngf*multin, self.ngf*multout, 4, 2, 1, 0, bias=bias))]
              else:  # use larger kernel for topmost layers
                layers = [("deconv"+str(index), nn.ConvTranspose2d(self.ngf*multin, self.ngf*multout, 6, 2, 2, 0, bias=bias))]

            else:  # using deconv for lower and this is a higher layer so using upsample+conv 
              layers = [ ("upsample"+str(index), nn.Upsample(scale_factor=2, mode='nearest')),
                           ("replpad"+str(index), nn.ReplicationPad2d(1)), 
                           ("conv"+str(index), nn.Conv2d(self.ngf*multin, self.ngf*multout, 3, 1, 0, bias=bias))]

            if index in opt.attn_ids:
                layers.append(("attn"+str(index), Self_Attn(self.ngf * multout, monitor=monitor)))

            if (normL not in [PixelNormalization, None]):
                layers.append(("norm"+str(index), normL(self.ngf * multout)))

            #layers.append(("relu"+str(index), self.activation)) 
            if (normL == PixelNormalization):
                layers.append(("norm"+str(index), normL(self.ngf * multout)))

            if index in opt.median_ids and opt.median > 0:
                median_layer = ("median"+str(index), MedianPool2d(kernel_size=opt.median, stride=opt.medianstride, same=True)) 
                layers.insert(0, median_layer)
                if opt.medianstride > 1:
                    resize_layer = ("medianresize"+str(index), nn.Upsample(scale_factor=opt.medianstride, mode='nearest'))
                    layers.insert(1, resize_layer)

            #if (opt.dropout >0) and index in opt.dropout_ids:
            #    layers.append(("dropout"+str(index), nn.Dropout2d(opt.dropout))) 

            return layers
            
        class UpBlock(nn.Module):   
            
             def __init__(self, index, multin, multout, normL, activation, bias, residual=False, ngf=0, monitor=False, noise = 0):
                 super(UpBlock, self).__init__()
                 layers = upsample_level(index, multin, multout, normL, bias, monitor)
                 self.blocks = nn.Sequential(OrderedDict(layers))
                 self.residual = residual
                 self.activation = activation
                 self.noise = noise      
                 if residual:
                     self.shortcut = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(ngf*multin, ngf*multout, 1, 1, 0, bias=bias))              
                 
             def forward(self, x):
                 #print(self.activation)
                 if self.noise > 0:
                     x = x + x.new(x.shape).normal_(0, self.noise)
                 if self.residual:
                     x = self.activation(self.shortcut(x) + self.blocks(x))
                 else:
                     x = self.activation(self.blocks(x))
                 return x  
                                  
 
        if self.nolin:
            if (norm_layer is None):
                layers2 = [("inrelu", self.activation)]
            elif (norm_layer != PixelNormalization):
                layers2 = [("innorm", norm_layer(self.ngf * 8)),
                           ("inrelu", self.activation)]
            else:
                layers2 = [("inrelu", self.activation),
                           ("innorm", norm_layer(self.ngf * 8))]



            #if opt.upsample:
            #  layers = [("in-upsample", nn.Upsample(scale_factor=2, mode='bilinear')),
            #            ("inconv", nn.Conv2d(self.nz, self.ngf*8, 3, 1, 1, bias=self.use_bias))]
            #else:
            layers = [("inlayer", nn.ConvTranspose2d(     self.nz, self.ngf*8, 4, 2, 1, bias=self.use_bias))]

            layers.extend(layers2)
            
        else:
            layers = [("inlayer", nn.Linear(self.nz, self.ngf*8*4)),
                     ("inrelu", self.activation),
                     ("inview", Zview())]

        #layers = [("inlayer", nn.Linear(self.nz, self.ngf*8*4)),
        #          ("inview", Zview())]
                          
        upsampleL = []
        outputSize = 2
        multin = 8

        n512 = n512 - opt.xlayers
        print(self.size, upsample_steps, n512)  
        for i in range(0, upsample_steps):
            outputSize = outputSize * 2 
            if i > n512 and multin > 1:
                multout = int(multin/2)
            else:
                multout = multin                
            index = i + 1
            noise = 0
            print(i,index, outputSize, multin, multout)
            if i in opt.noise_ids:
                noise = opt.layer_noise
            upsampleL.append(("upblock"+str(index), UpBlock(i+1, multin, multout, norm_layer, self.activation, self.use_bias, residual=opt.residual, ngf=self.ngf, monitor=opt.monitor, noise=noise)))
            multin = multout

        if opt.x2:
            upsampleL.append(("upblock"+str(i + 2), UpBlock(i + 2, multin, multin, norm_layer,  self.activation, self.use_bias, residual=opt.residual, ngf=self.ngf, monitor=opt.monitor)))       
        elif opt.x4:
            upsampleL.append(("upblock"+str(i + 2), UpBlock(i + 2, multin, multin, norm_layer,  self.activation, self.use_bias, residual=opt.residual, ngf=self.ngf, monitor=opt.monitor)))        
            upsampleL.append(("upblock"+str(i + 3), UpBlock(i + 3, multin, multin, norm_layer,  self.activation, self.use_bias, residual=opt.residual, ngf=self.ngf, monitor=opt.monitor)))        

        #print(upsampleL)
        layers.extend(upsampleL)

        if opt.hardtanh:
            final = [
                #("outconv", nn.ConvTranspose2d(self.ngf, self.nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias)),
                ("outconv", nn.Conv2d(self.ngf, self.nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias)),
                ("outactiv", nn.Hardtanh()) ]
        else:
            final = [
                #("outconv", nn.ConvTranspose2d(self.ngf, self.nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias)),
                ("outconv", nn.Conv2d(self.ngf, self.nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias)),
                ("outactiv", nn.Tanh()) ]
            
        layers.extend(final)

        if opt.medianout > 0:
            layers.extend([("outmedian", MedianPool2d(kernel_size=opt.medianout, stride=1, same=True))])

        print(layers)
        ol = OrderedDict(layers)
        print(ol)
        self.main = nn.Sequential(ol)
        
    def forward(self, input):
        if not self.nolin:
            input = input.view(-1, self.nz)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            #print("G", output.shape)
        return output


# discriminator

class _netD(nn.Module):
    def __init__(self, ngpu=1, use_sigmoid=True, norm_layer=nn.BatchNorm2d, opt=None):
        super(_netD, self).__init__()
        self.nz = opt.nz
        self.nc = opt.nc
        self.ndf = opt.ndf
        self.D2 = opt.D2 is not None 
        if opt.Dsize == 0:
            self.size = opt.imageSize
        else:
            self.size = opt.Dsize
        self.ngpu = ngpu
        #if norm_layer == nn.BatchNorm2d:
        #    norm_layer = nn.BatchNorm2d(eps = opt.bneps)
        #    print("******************", norm_layer)
        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func==nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer==nn.InstanceNorm2d
        self.activation = nn.ReLU(inplace=True)
        if opt.elu:
            self.activation = nn.ELU(inplace=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        if opt.elu:
            self.activation = nn.ELU(inplace=True)
        elif opt.sinerelu:
            self.activation = SineReLU()    
                  
  
        print("defining _netD")

        start_idx = 0 #- opt.xlayers
        if opt.x2:
            self.size = int(self.size/2)
            #start_idx = 1 
        elif opt.x4:
            self.size = int(self.size/4)
            #start_idx = 1 

        downsample_steps = int(round(math.log(self.size, 2))) - 3 - opt.xlayers
        n512 = downsample_steps - 3 #+ opt.xlayers
        dnsampleL = []
        inputSize = int(self.size) # note that the first convlayer above these already downsamples by 2
        multin = 1

        def downsample_level(index, multin, multout, normL, bias, monitor):
            if opt.D2:
                layers = [("conv"+str(index), nn.Conv2d(self.ndf*multin, self.ndf*multout, 4, 2, 1, bias=self.use_bias)),
                          ("convb"+str(index), nn.Conv2d(self.ndf*multout, self.ndf*multout, 3, 1, 1, bias=self.use_bias))]
            else:
                layers = [("conv"+str(index), nn.Conv2d(self.ndf*multin, self.ndf*multout, 4, 2, 1, bias=self.use_bias))]

            if index in opt.attnD_ids:
                layers.append(("attn"+str(index), Self_Attn(self.ndf * multout, monitor=monitor)))
            
            if opt.nonlin1st:
                layers.append(("relu"+str(index), self.activation))
  
            if normL == nn.BatchNorm2d:
                layers.append(("norm"+str(index), normL(self.ndf * multout, eps=opt.bneps)))
            else:
                layers.append(("norm"+str(index), normL(self.ndf * multout)))

            if not opt.nonlin1st:
                layers.append(("relu"+str(index), self.activation))
            
            if index in opt.dmedian_ids and opt.dmedian > 0:
                median_layer = ("median"+str(index), MedianPool2d(kernel_size=opt.dmedian, stride=1, same=True)) 
                layers.insert(0, median_layer)

            if (opt.dropout >0) and index in opt.dropout_ids:
                layers.append(("dropout"+str(index), nn.Dropout2d(opt.dropout))) 

            return layers

        if norm_layer == nn.BatchNorm2d:
            norm0 = ("norm0", norm_layer(self.ndf, eps=opt.bneps))
            norm0b = ("norm0b", norm_layer(self.ndf, eps=opt.bneps))
        else:
            norm0 = ("norm0", norm_layer(self.ndf))   
            norm0b = ("norm0b", norm_layer(self.ndf))   


        if opt.nonlin1st:
          if opt.x2:
            layers = [("inconv", nn.Conv2d(self.nc, self.ndf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)),
                      ("inrelu", self.activation),
                      ("conv0", nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=self.use_bias)),
                      ("relu0", self.activation),
                      norm0
                      ]
          elif opt.x4:
            layers = [("inconv", nn.Conv2d(self.nc, self.ndf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)),
                      ("inrelu", self.activation),
                      ("conv0", nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=self.use_bias)),
                      ("relu0", self.activation),
                      norm0,
                      ("conv0b", nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=self.use_bias)),
                      ("relu0b", self.activation),
                      norm0b
                      ]
          else:
            layers = [("inconv", nn.Conv2d(self.nc, self.ndf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)),
                      ("inrelu", self.activation)]
        else:
          if opt.x2:
            layers = [("inconv", nn.Conv2d(self.nc, self.ndf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)),
                      ("inrelu", self.activation),
                      ("conv0", nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=self.use_bias)),
                      norm0,
                      ("relu0", self.activation)
                      ]
          elif opt.x4:
            layers = [("inconv", nn.Conv2d(self.nc, self.ndf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)),
                      ("inrelu", self.activation),
                      ("conv0", nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=self.use_bias)),
                      norm0,
                      ("relu0", self.activation),
                      ("conv0b", nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=self.use_bias)),
                      norm0b,
                      ("relu0b", self.activation)
                      ]
          else:
            layers = [("inconv", nn.Conv2d(self.nc, self.ndf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)),
                      ("inrelu", self.activation)]

        if opt.dmedianin > 0:
                median_layer = ("medianin", MedianPool2d(kernel_size=opt.dmedianin, stride=1, same=True)) 
                layers.insert(0, median_layer)


        suffix = ['', 'b', 'c', 'd', 'e']

        print(layers)
        print(self.size, downsample_steps, n512, start_idx)
        if opt.xlayers > 0:
          for i in range(0, opt.xlayers):
            inputSize = int(inputSize / 2) 
            index = '0' + suffix[i]
            print(i, index, multin, multin, inputSize)  
            dnsampleL.extend(downsample_level(index, multin, multin, norm_layer, self.use_bias, opt.monitor))
  
        for i in range(start_idx, downsample_steps+start_idx):
            inputSize = int(inputSize / 2) 
            if multin < 8:
                 multout = multin*2
            else:
                 multout = multin               
            index = i + 1
            print(i, index, multin, multout, inputSize)  
            dnsampleL.extend(downsample_level(index, multin, multout, norm_layer, self.use_bias, opt.monitor))
            multin = multout

        layers.extend(dnsampleL)

        corr = 0
        g = 4 if opt.batchSize % 4 == 0 else 1

        if opt.minibatchstd:
            mbstdlayer = [("batchstd", BatchStdConcat(groupSize=g))]
            layers.extend(mbstdlayer)
            corr = 1

        if opt.hgan:
            final = [ ('outconvA', nn.Conv2d(self.ndf*multin + corr, 1, kernel_size=6, stride=4, padding=1, bias=self.use_bias))] #,
        elif opt.hgan2:
            final = [ ('outconvA', nn.Conv2d(self.ndf*multin, 1, kernel_size=4, stride=2, padding=1, bias=self.use_bias)),
                      ('outconvB', nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=self.use_bias))] 
        else:
            final = [ ('outconvA', nn.Conv2d(self.ndf*multin + corr, self.ndf*multin + corr, kernel_size=4, stride=2, padding=1, bias=self.use_bias)),
                ('outconvB', nn.Conv2d(self.ndf*multin + corr, 1, kernel_size=4, stride=2, padding=1, bias=self.use_bias))] 

        layers.extend(final)
        if use_sigmoid:
            layers += [("outactiv", nn.Sigmoid())]
        #print(layers)
        self.main = nn.Sequential(OrderedDict(layers))

    def forward(self, input):
        #print("Din",input.size())
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        #print("D1", output.size())
        output = output.view(-1, 1).squeeze(1)
        #print("D2", output.size())
        return output

# encoder, duplicates my mygan architecture as of 4/2017  

class _netE(nn.Module):
    def __init__(self, ngpu=1, use_sigmoid=True, norm_layer=nn.BatchNorm2d, opt=None):
        super(_netE, self).__init__()
        self.ngpu = ngpu
        self.nz = opt.nz
        self.nc = opt.nc
        self.nef = opt.nef
        self.size = opt.imageSize
        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func==nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer==nn.InstanceNorm2d
            
        downsample_steps = int(round(math.log(self.size, 2))) - 2
        #n512 = downsample_steps - 4
        print("defining _netE")

        def downsample_level(index, multin, multout, normL, bias, leaky=True):
            layers = [("conv"+str(index)+"a", nn.Conv2d(self.nef*multin, self.nef*multout, kernel_size=3, stride=2, padding=1, bias=self.use_bias)),
            ("conv"+str(index)+"b", nn.Conv2d(self.nef*multout, self.nef*multout, kernel_size=3, stride=1, padding=1, bias=self.use_bias)),
            ("norm"+str(index), norm_layer(self.nef*multout))]
            if leaky:
                layers.append(("relu"+str(index), nn.LeakyReLU(0.2, inplace=True)))
            else:
                layers.append(("relu"+str(index), nn.ReLU(inplace=True)))
            #print(layers)

            return layers

        sequence = [
            ("inconv", nn.Conv2d(self.nc, self.nef, kernel_size=3, stride=2, padding=1, bias=self.use_bias)),
            ("inrelu", nn.LeakyReLU(0.2, inplace=True))]

        downsampleL = []
        currentSize = int(self.size/2)
        multin = 1
         
        print(self.size, downsample_steps)  
        for i in range(0, downsample_steps):
             currentSize = int(currentSize * 2) 
             if multin < 8:
                 multout = multin*2
             else:
                 multout = multin                
             index = i + 1
             leaky = index != downsample_steps
             print(i, currentSize, multin, multout, leaky)
             downsampleL.extend(downsample_level(i+1, multin, multout, norm_layer, self.use_bias, leaky=leaky))
             multin = multout

        sequence.extend(downsampleL)
        
        final = [ ("outview", ZEview()), ("outlayer", nn.Linear(self.nef*8*4, self.nz))]
        sequence.extend(final)
        #sequence += [("outtanh", nn.Tanh())]
        
        #if use_sigmoid:
        #    sequence += [("outactiv", nn.Sigmoid())]
            
            
        #print(sequence)    
        self.main = nn.Sequential(OrderedDict(sequence))

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        #output = output/output.std()
        #output = output - output.mean()
        return output.view(-1, self.nz) # KORJATTU 25.7.2018 !!!
        

