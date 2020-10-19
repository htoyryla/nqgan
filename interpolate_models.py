import argparse
import os
import torch
import torch.nn as nn
#import torchvision
#import random

parser = argparse.ArgumentParser()
parser.add_argument('--m1', default=None, type=str, help='model 1')
parser.add_argument('--m2', default=None, type=str, help='model 2')
parser.add_argument('--output', default='frames', type=str, help='output model')
parser.add_argument('--beta', type=float, default=0.5, help='interpolation factor')

opt = parser.parse_args()


print("loading: ",opt.m1,opt.m2)
p1 = torch.load(opt.m1)
p2 = torch.load(opt.m2)

names1 = p1.keys()
names2 = p2.keys()
p = {}  

# calculate weighted averages of two two models
beta = opt.beta
for name in names1:
        if name in names2:
            print("1:"+name+" - 2:"+name, beta, 1-beta)
            p[name] = beta*p1[name].data + (1-beta)*p2[name].data

print("saving to ", opt.output)    
torch.save(p, opt.output)


