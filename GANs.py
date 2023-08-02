
# Implementation of Deep Convoultional GANs

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# initialize the hyper parameters:

batch_size = 64
image_size = 64

transform = transforms.Compose([transforms.Scale(image_size), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),])

# Loading the Dataset
dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Defining the Weight Init function that initializes all the values
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        m.weight.data.normal(0.0,0.02)
    elif classname.find('BatchNorm') !=-1:
        m.bias.data.fill_(0)

# Defining the Generator

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Now Creating the Meta Module which is a sequence of Several Modules

        self.main = nn.Sequential(
            # Layer 1: Inverse Convolution Layer
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),

            # Layer 2: Batch Normalization Layer
            nn.BatchNorm2d(512),
            
            #Layer 3: Rectilinear Activation Function
            nn.Relu(True),

            # Layer 4: Inverse Convolutional Layer
            nn.ConvTranspose2d(512, 256, 4,2,1, batch_size=False),

            # Layer 5: Inverse Convolutional Layer
            nn.BatchNorm2d(256)

        )
