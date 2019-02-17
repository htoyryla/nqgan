import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from nqmodel4 import _netG

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline', type=str, help='trained model name')
parser.add_argument('--which_epoch', default='24', type=str, help='0,1,2,3,4...')
parser.add_argument('--batchSize', default=4, type=int, help='batchsize')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--imageSize', default=64, type=int, help='image size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--nolin', action='store_true', help='use some linear layers in models')
parser.add_argument('--instance', action='store_true', help='use instance norm')
parser.add_argument('--batchnorm', action='store_true', help='use batch norm')
parser.add_argument('--upsample', action='store_true', help='use upsample and conv instead of convtranspose')

parser.add_argument('--howMany', default=100, type=int, help='how many transitions')
parser.add_argument('--steps', default=20, type=int, help='interpolation steps')
parser.add_argument('--median', type=int, default=0, help='median layer kernel, set 0 for no median filter')
parser.add_argument('--medianindex', type=str, default="", help='')
parser.add_argument('--medianout', type=int, default=0, help='')
parser.add_argument('--size', type=int, default=2, help='')
parser.add_argument('--mean', type=float, default=0, help='')
parser.add_argument('--std', type=float, default=1, help='')
parser.add_argument('--output', default='frames', type=str, help='output directory')


opt = parser.parse_args()
opt.debug = False
#opt.nolin = True
#opt.size = 2
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

    cudnn.benchmark = True #opt.cudnn #True
    cudnn.enabled = True

str_median = opt.medianindex.split(',')
median_ids = []
for mid in str_median:
    if (mid == ""): continue
    mid = int(mid)
    if mid>=0:
        median_ids.append(mid)
opt.median_ids = median_ids

modeldir = outdir = os.path.join('runs',opt.name,'model')
outdir = os.path.join('runs',opt.name,opt.output)
print(outdir)

def makeZ():
    noise = torch.FloatTensor(opt.batchSize, opt.nz, opt.size, opt.size).normal_(opt.mean,opt.std)
    #m = nn.ZeroPad2d(8)
    #noise = m(noise)
    return noise

#---------------generate images
def generate_img(model):
    ctr = 0
    start_noise = makeZ() #torch.FloatTensor(opt.batchSize, opt.nz, 2, 2).normal_(0,1)
    start_noise = start_noise.cuda()
    #start_noise = Variable(start_noise)
    for i in range(opt.howMany):
        end_noise = makeZ() #torch.FloatTensor(opt.batchSize, opt.nz, 2, 2).normal_(0,1)
        end_noise = end_noise.cuda()
        #end_noise = Variable(end_noise)
        delta = (end_noise - start_noise)/opt.steps 
        for j in range (opt.steps):
          fake = model(start_noise.detach() + j * delta)
          #fake = outputs.data
          print(ctr)
          im = fake[0,:,:,:]
          #im = im/2 + 0.5
          torchvision.utils.save_image(
                    im.view(1, im.size(0), im.size(1), im.size(2))[0],
                    os.path.join(outdir, 'frame-%d_%d.png'%(int(opt.which_epoch),ctr)),
                    nrow=1,
                    padding=0,
                    normalize=True)
          del fake
          del im
          ctr = ctr + 1            
        start_noise = end_noise        

#-----------------Load model
def load_network(network):
    save_path = os.path.join(modeldir,'netG_epoch_%s.pth'%opt.which_epoch)
    p = torch.load(save_path)
    p2 = {}
    for k in p.keys():
        k2 = "main."+k
        p2[k2] = p[k]
    network.load_state_dict(p)
    return network

if opt.instance:
    netG = _netG(norm_layer=nn.InstanceNorm2d, opt=opt)
elif opt.batchnorm:
    netG = _netG(norm_layer=nn.BatchNorm2d, opt=opt)
else:
    netG = _netG(opt=opt)
print(netG)
model = load_network(netG)
model = model.cuda()
model.eval()

generate_img(model)
