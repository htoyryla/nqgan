
import argparse
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='./datasets', help='path to datasets folder')
parser.add_argument('--name', required=True, help='project name')

opt = parser.parse_args()
dataset_root = opt.dataset_root #'datasets/'
dataset_name = opt.name # 'o512'

dataset = datasets.ImageFolder(os.path.join(dataset_root, dataset_name), transform=transforms.Compose([transforms.Resize((256, 256)),
                             transforms.ToTensor()]))

loader = torch.utils.data.DataLoader(dataset,
                         batch_size=16,
                         num_workers=0,
                         shuffle=False)

mean = 0.0
for images, _ in loader:
    batch_samples = images.size(0) 
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(loader.dataset)

var = 0.0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(loader.dataset)*256*256))

print (dataset_name, mean, std)

output = {'name':dataset_name, 'mean':mean.tolist(), 'std':std.tolist()}

fn = os.path.join(dataset_root,dataset_name+'_meta.json')
with open(fn, 'w') as outfile:
    json.dump(output, outfile)
    print('dsmeta save to ',fn)
