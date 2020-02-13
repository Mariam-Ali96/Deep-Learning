import os
import torch
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms as transforms




#%% Define paths

data_root_dir = '../datasets'


#%% Create dataset

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = MNIST(data_root_dir, train=True,  download=True, transform=train_transform)
test_dataset  = MNIST(data_root_dir, train=False, download=True, transform=test_transform)

### Plot some sample
plt.close('all')
fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    img, label = random.choice(train_dataset)
    ax.imshow(img.squeeze().numpy(), cmap='gist_gray')
    ax.set_title('Label: %d' % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()


num_workers = 0
batch_size=20

#########################################################################
##########################Transpose AutoEncoder#########################
########################################################################
#Input data is passed through an encoder
#Input images are 28x28x1 in size, images will be passed through encoder layers
#Encoder will compress the input
#It will have convolutional layers followed by max pooling layer to reduce dimensions

#Compressed data is is passed through a decoder to reconstruct the input data
#This layer will bring back to original dimension 28x28x1
#Will use ##transposed convolutional layers## to increase width and height of compressed input
#Transpose convolution layers can lead to artifacts
# in the final images, such as checkerboard patterns.
#This is due to overlap in the kernels..

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))

        return x

# initialize the NN
model1 = ConvAutoencoder()
print(model1)


### Define dataloader
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

llossT={}

# Loss function
criterion = nn.MSELoss()
noise_factor=0.5

# Optimizer
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)

epochs = 20
for e in range(1, epochs+1):
    train_loss1 = 0.0  # monitor training loss

    ###################
    # train the model #
    ###################
    for data in train_loader:
        images, _ = data                        # we are just intrested in just images
        # no need to flatten images
        optimizer1.zero_grad()                   # clear the gradients
        outputs = model1(images)                 # forward pass: compute predicted outputs
        loss1 = criterion(outputs, images)       # calculate the loss
        loss1.backward()                         # backward pass
        optimizer1.step()                        # perform optimization step
        train_loss1 += loss1.item()*images.size(0)# update running training loss

    # print avg training statistics
    train_loss1 = train_loss1/len(train_loader)
    print('Epoch: {}'.format(e),
          '\tTraining Loss: {:.4f}'.format(train_loss1))


    llossT[e]=float(train_loss1)

time1= list(llossT.keys())
error1= list(llossT.values())

fig= plt.figure(figsize=(6,8))
plt.title('Loss Plot')
#plt.plot(time1,error, color='red')
plt.plot(time1, error1, color='green')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

# Lets get batch of test images
# Lets get batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

output = model1(images)                     # get sample outputs
images = images.numpy()                    # prep images for display
output = output.view(batch_size, 1, 28, 28)# resizing output
output = output.detach().numpy()           # use detach when it's an output that requires_grad

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)

    plt.tight_layout()
    plt.show()


######################################################################
import Denoiser
# initialize the NN
model = Denoiser.DeNoiser()
print(model)

### Define dataloader
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
lloss={}
# Loss function
criterion = nn.MSELoss()

# Optimizer
noise_factor=0.5

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for adding noise to images
noise_factor=0.5
epochs = 20
for e in range(1, epochs+1):
    train_loss = 0.0  # monitor training loss

    ###################
    # train the model #
    ###################
    for data in train_loader:
        images, _ = data                        # we are just intrested in images
        # no need to flatten images
        ## add random noise to the input images
        noisy_imgs = images + noise_factor * torch.randn(*images.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        optimizer.zero_grad()                   # clear the gradients
        outputs = model(noisy_imgs)             # forward pass: compute predicted outputs
        loss = criterion(outputs, images)       # calculate the loss
        loss.backward()                         # backward pass
        optimizer.step()                        # perform optimization step
        train_loss += loss.item()*images.size(0)# update running training loss

    # print avg training statistics
    train_loss = train_loss/len(train_loader)
    print('Epoch: {}'.format(e), '\tTraining Loss: {:.4f}'.format(train_loss))
    lloss[e]=float(train_loss)

time= list(lloss.keys())
error= list(lloss.values())

fig= plt.figure(figsize=(6,8))
plt.title('Loss Plot Denoiser vs AutoEncoder')
plt.plot(time, error, color='green', label='Denoiser loss')
plt.plot(time1,error1, color='red', label= 'AutoEncoder loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Denoiser vs AutoEncoder')
plt.show()

##############################################################
##Comparing inputs before encoding and output after decoding##
##############################################################

# Lets get batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# add noise to the test images
noisy_imgs = images + noise_factor * torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

output = model(noisy_imgs)                 # get sample outputs
noisy_imgs = noisy_imgs.numpy()            # prep images for display
output = output.view(batch_size, 1, 28, 28)# resizing output
output = output.detach().numpy()           # use detach when it's an output that requires_grad

# plot the first ten input images and then reconstructed images
fig1, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
# input images on top row, reconstructions on bottom
fig1.suptitle('Comparing inputs before encoding and output after decoding')
for noisy_imgs, row in zip([noisy_imgs, output], axes):
    for img, ax in zip(noisy_imgs, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
    plt.tight_layout()
    plt.show()
