
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import json


dataset_root = 'datasets/'
dataset_name = 'o512'

dataset = datasets.ImageFolder(dataset_root+dataset_name, transform=transforms.Compose([transforms.Resize((256, 256)),
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

with open(dataset_root+dataset_name+'_meta.json', 'w') as outfile:
    json.dump(output, outfile)
