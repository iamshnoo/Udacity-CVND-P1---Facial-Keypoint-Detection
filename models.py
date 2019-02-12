## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs      
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel       
        ## input image size is 224 * 224 pixels        
        # first convolutional layer
        ## (W-F)/S + 1 = (224-5)/1 + 1 = 220
        ## self.conv1 = nn.Conv2d(1, 32, 5) # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        
        # NOTE : This netork architecture is largely based on the LeNet Architecture with some modifications 
        #        to suit the needs of this project and the constraints of my hardware.
       
        # input image size is 96 * 96 pixels
        
        # first convolutional layer
        self.conv1 = torch.nn.Conv2d(1,32,5) # (32,92,92) output tensor # (W-F)/S + 1 = (96-5)/1 + 1 = 92
        
        # first Max-pooling layer
        self.pool1 = torch.nn.MaxPool2d(2,2) # (32,46,46) output tensor
        
        # second convolutional layer
        self.conv2 = torch.nn.Conv2d(32,64,5) # (64,44,44) output tensor # (W-F)/S + 1 = (46-5)/1 + 1 = 42
        
        # second Max-pooling layer
        self.pool2 = torch.nn.MaxPool2d(2,2) # (64,21,21) output tensor
        
        # Fully connected layer
        self.fc1 = torch.nn.Linear(64*21*21, 1000)   
        self.fc2 = torch.nn.Linear(1000, 500)       
        self.fc3 = torch.nn.Linear(500, 136)        
        self.drop1 = nn.Dropout(p=0.4)
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
          
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop1(x)
      
        # Flatten before passing to the fully-connected layers.
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

