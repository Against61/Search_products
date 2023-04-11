from torchvision import models
import timm
import torch
import torch.nn as nn

#external import
from train.ArcFace import ArcFaceHead


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Download the pre-trained resnet50 model
        self.resnet = timm.create_model('resnet50', pretrained=True)
        addition_head = ArcFaceHead(512, 11319)

        # Create an extra head
        self.resnet.fc = nn.Linear(2048, 512)
        #         self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.archead = addition_head

    def forward(self, x, labels):
        # Go through the resnet50 model
        x = self.resnet(x)
        #         x = self.fc1(x)
        x = self.archead(x, labels)

        return x