# nqgan

A versatile GAN supporting many different options.

## Installation

Basic requirements: python >=3.6, pytorch >= 0.41, opencv >= 3.0

## Datasets

Use any images you want. Create a directory e.g. datasets/flowers/train and plcae your images there. It is usually better to resize or crop images to the largest size you want to train, for instance 512x512, but this is no strictly necessary, it can also be done at runtime.

## Training

###Basic example. 
```
python nqgan4.py --dataroot datasets/flowers/ --dataset folder --name test001 --imageSize 128 --batchSize 128 --withoutE --init dcgan --rsgan --lrD 0.0002 
```


dataset:  path to your dataset
name:     name you want to use for this experiment, files will be places in runs/<name>
imageSize: your choice from 64, 128, 256, 512, 1024, larger size will require more memory, adjust if you run out of memory
batchSize: with smaller image size you can use larger batch size to speed up training, with 512px you can try batch size of 8 as a starting point; adjust if you run out of memory
withoutE: train as a normal GAN; when left out train also with an encoder
init: dcgan - init weights as for dcgan, def - init weight using pytorch defaults
rsgan: use relativistic GAN, usually helps with smaller datasets
lrD: learning rate for discriminator, default is 0.00005 but often 0.0002 works better with rsgan 
```
###How to know how training works

Look at D(x) and D(G(z)) values. D(x) should increase close to 1, this indicates that discriminator is identifying real images correctly.

D(G(z)), can decrease close to zero, meaning that generator is not good enough to fool the discriminator. This is normal. However, if D(G(z)) gets down to zero and stays there, generator is no longer learning.

###How to help training

Adjust learning rates: lrD for discriminator, lrG for generator.

Often weights can grow unbalanced, so that some parts of a model dominate. This can be prevented by clipping weights, for instance
--clipD 0.02 if discriminator is too strong or behaving oddly. --clipG 0.02 does the same for the generator. You can also try higher values.

###Other training related parameters:

niter: how many iterations to run (one iteration traverses the whole dataset once, so usually with a small dataset you need a larger niter)
step: decrease learning rate after <step> iterations (normal practice in training neural networks)
gamma: a multiplier used to reduce learning rate, 1 means no reduction, 0.9 means 10% reduction, 0.5 50% reduction.


  
