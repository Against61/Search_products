import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#external imports
from train.train import train_model
from train.model import MyModel
from train.Dataloader import SopDataset
from train.ArcFace import ArcFaceHead

#Parameters:
epochs = 1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)

#Train the model
train_model(model, criterion, optimizer, scheduler, epochs)