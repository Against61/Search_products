import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

#external imports
from train.train import train_model
from train.model import MyModel
from train.Dataloader import SopDataset
from train.ArcFace import ArcFaceHead

# Define argparser and arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
#add parser argument to start training proccess
argparser.add_argument('--train', type=bool, default=False, help='Start training proccess')


# Parse arguments
args = argparser.parse_args()

# Set epochs to the value specified by the user
epochs = args.epochs

# Set up the device and the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)

# Train the model

__main__ = "main"

if args.train:
    train_model(model, criterion, optimizer, scheduler, epochs)
else:
    print("Training is not enabled. Please enable training by setting --train=True")