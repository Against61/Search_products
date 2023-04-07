from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import lr_scheduler
import torch.optim as optim
import torch
from torch import autograd
from pytorch_metric_learning import losses
from torch.utils.data.dataloader import DataLoader, RandomSampler
import torch.nn as nn
import numpy as np


#external_import
from augmentation import train_transform, val_transform
from model import MyModel
from Dataloader import SopDataset
from ArcFace import ArcFaceHead

# Parameters:
B_size = 16
epochs = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MyModel()
model.to(device)

# Data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# train_transform = A.Compose(
#     [
#         A.Resize(height=256, width=256),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.CenterCrop(p=1, height=224, width=224),
#         A.OneOf([
#                 A.ElasticTransform(border_mode=1),
#                 A.GridDistortion(p=1),
#                 A.GaussNoise(p=1),
#                 A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
#         ], p=1.0),
#         A.OneOf([
#                 A.ElasticTransform(border_mode=1),
#                 A.GridDistortion(p=1),
#                 A.GaussNoise(p=1),
#                 A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
#         ], p=1.0),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]
# )
#
# val_transform = A.Compose(
#     [
#         A.Resize(height=256, width=256),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]
# )

train_dataset = SopDataset(transform=train_transform())
val_dataset = SopDataset(transform=val_transform())

# Preparing and dividing data for learning process
trainloader = DataLoader(train_dataset, batch_size=B_size, shuffle=True)
sampler_test = RandomSampler(trainloader)
testloader = DataLoader(train_dataset, sampler = sampler_test , batch_size=B_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
criterion.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-3, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.00005)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)
# eff_net_wts = copy.deepcopy(eff_net.state_dict())



min_valid_loss = np.inf

for e in range(epochs):
    train_loss = 0.0
    model.train()  # Optional when not using Model Specific layer
    for data, labels in trainloader:
        data, labels = data.to(device), labels.to(device)
        data = torch.tensor(data, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        optimizer.zero_grad()
        target = model(data, labels)
        loss = criterion(target, labels)
        # writer.add_scalar('train/loss', train_loss, e)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    with torch.no_grad():
        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            data = torch.tensor(data, dtype=torch.float)
            target = model(data, labels)
            loss = criterion(target, labels)
            valid_loss = loss.item() * data.size(0)

    print(
        f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(testloader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'checkpoint/saved_model.pth')
# writer.flush()
