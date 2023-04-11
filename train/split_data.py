from torch.utils.data.dataloader import DataLoader, RandomSampler

#external import
from train.Dataloader import SopDataset
from train.augmentation import train_transform, val_transform

#parameters:
B_size = 16

#data augmentation
train_dataset = SopDataset(transform=train_transform())
val_dataset = SopDataset(transform=val_transform())

# Preparing and dividing data for learning process
trainloader = DataLoader(train_dataset, batch_size=B_size, shuffle=True)
sampler_test = RandomSampler(trainloader)
testloader = DataLoader(train_dataset, sampler = sampler_test , batch_size=B_size, shuffle=False)
