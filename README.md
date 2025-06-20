# nqgan

A versatile GAN supporting many different options.

**NOTE: This repository is made public as a media archeological record only. I started developing nqgan in Spring 2018 for my artistic work and it gradually grew into a monster with a huge number of options. It was my primary artistic tool until the end of 2020.
The code has evolved for the purposes of my personal artistic work, and there has never been any attempt to clean up the code or refactor it. So be prepared to find it a mess. Also, it may not function
correctly from theoretical point of view. It never mattered much to me. Whatever has produced artistically useful results has been ok for me. Nevertheless, a few other media artists have managed to use this
succesfully and produce works of art succesfully**

**There will absolutely not be any further develepoment or maintenance here!!!!**


## Installation

Basic requirements: python >=3.6, pytorch >= 0.41, opencv >= 3.0

## Datasets

Use any images you want. Create a directory e.g. datasets/flowers/train and plcae your images there. It is usually better to resize or crop images to the largest size you want to train, for instance 512x512, but this is no strictly necessary, it can also be done at runtime.

## Training

### Basic example. 
```
python nqgan4.py --dataroot datasets/flowers/ --dataset folder --name test001 --imageSize 128 --batchSize 128 --withoutE --init dcgan --rsgan --lrD 0.0002 
```
Options used:
```
dataroot:  path to your dataset
name:     name you want to use for this experiment, files will be places in runs/<name>
imageSize: your choice from 64, 128, 256, 512, 1024, larger size will require more memory, adjust if you run out of memory
batchSize: with smaller image size you can use larger batch size to speed up training, with 512px you can try batch size of 8 as a starting point; adjust if you run out of memory
withoutE: train as a normal GAN; when left out train also with an encoder
init: dcgan - init weights as for dcgan, def - init weight using pytorch defaults
rsgan: use relativistic GAN, usually helps with smaller datasets
lrD: learning rate for discriminator, default is 0.00005 but often 0.0002 works better with rsgan 
```

### How to know how training works

Look at D(x) and D(G(z)) values. D(x) should increase close to 1, this indicates that discriminator is identifying real images correctly.
```
D(G(z)), can decrease close to zero, meaning that generator is not good enough to fool the discriminator. 
This is normal. However, if D(G(z)) gets down to zero and stays there, generator is no longer learning.
```
You can also have a look at sample images generated during training in /runs/<name>/visual

### How to help training

Adjust learning rates: lrD for discriminator, lrG for generator.

Often weights can grow unbalanced, so that some parts of a model dominate. This can be prevented by clipping weights, for instance
--clipD 0.02 if discriminator is too strong or behaving oddly. --clipG 0.02 does the same for the generator. You can also try higher values.

### Other training related parameters:
```
niter: how many iterations to run (one iteration traverses the whole dataset once, so usually with a small dataset you need a larger niter)
step: decrease learning rate after <step> iterations (normal practice in training neural networks)
gamma: a multiplier used to reduce learning rate, 1 means no reduction, 0.9 means 10% reduction, 0.5 50% reduction.
flip: training images are randomly flipped horizontally (so as to introduce more variety)
noise: add noise to training images, give st dev as parameter e.g. --noise 0.01
```
### Use cyclical learning rate

Cyclical learning rate can work better than the usual gradually decreasing rate.

Use for example 
'''
--lrD 0.0002 --lrG 0.0002 --cyclr 100
'''

to vary lr cyclically between 0.0002 and 0.000002.

I have also noticed that use of cyclical lr may reduce memory usage.

### Model architecture

By default, 

* generator uses pixel normalization, discriminator uses batch normalization
* generator uses linear layer at input
* generator uses transposed convolution
* both generator and discriminator use 64 * N channels

These can be changed

```
--instance use instance normalization in generator
--nonorm no normalization at all in generator
--nolin use a fully convolutional generator
--upsample use upsample and convolution instead of transposed convolution in generator
--ngf 128 adjust the number of channels in generator
--ndf 128 adjust the number of channels in discriminator
--nc 1 use one channel input (black and white) instead of 3 (color)
```

Higher number of channels uses more memory but also can produce higher quality

### Saving

Models will be saved at runs/name/model . Use --save_every to specify how ofter you want to save.

### Continue training
```
Use --netG path_to_saved_generator and --netD path_to_saved_discriminator to load pretrained
models instead of training from scratch. It is recommended to use a new name for the experiment, as in

--name test001 --netD runs/test000/model/netD_epoch_100.pth -netG runs/test000/model/netG_epoch_100.pth


Note that the model architecture parameters for the new should match
those used in the original training. Otherwise various errors are likely to occur.
They can be ignored though using --nostrict, then models will be initialized randomly
and then matching parts of the models copied, this can be used in special cases if 
you know what you are doing :)
```
### Other functionality

Nqgan has additional functionality not yet documented here. For instance one can add an encoder so that generator will be trained both according to the feedback from discriminator and according to how well the images are reproduced by the encoder-generator chain (as in an autoencoder).

By default, cudnn is disabled in this version. Use --cudnn to enable.

## How to use a trained model to generate images

There are several programs for using a trained model to generate images. For this to work, you need to have a saved model in runs/name/model and you need to know the options used in training. If in doubt about the options, look at the runs/name/cmd.txt file to see the command used for training.   

### Generate simple frames from a single iteration (aka epoch)

```
python nqframes4.py  --name nqtest --imageSize 512 --batchSize 4  --ndf 128 --howMany 10 --steps 1 --which_epoch 200
```

Name identifies the experiment you want to use, imageSize and any model architecture settings must match those used in training. The same does not apply to batchSize, you can even use 1 here to save memory. The only thing to note is that when using batch normalization in generator, results may be poor if a very small batch size is used here.

Which_epoch identifies which of the saved models you want to use. Have a look at the saved samples in runs/name/visual to select an iteration which you like. You might want to double check that you actually have a saved model from that iteration in runs/name/model (unless you saved at every iteration using --save_every 1).

howMany gives simply the number of images to be generated. Using steps higher than one will additionally generate morphs between images using the given number of steps. Thus, the total number of images generated will be howMany * steps.

You will find the images generated in runs/name/frames




  
