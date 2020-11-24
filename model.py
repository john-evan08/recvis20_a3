import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

nclasses = 20 

model = models.resnet152(pretrained=True)#pretrained on Imagenet
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, nclasses)
input_size = 224
print(model)
