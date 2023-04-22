import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import lr_scheduler
import torch.optim as optim
from torch import autograd
from pytorch_metric_learning import losses
import torch.nn as nn
import numpy as np

#external imports
from config import config
from train.model import MyModel
from train.Dataloader import SopDataset
from train.split_data import trainloader, testloader
from config.config import logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, epochs):
    criterion.to(device)
    model.to(device)
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
            logger.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            # Saving State Dict
            torch.save(model.state_dict(), config.CHECKPOINT_PATH_SAVE_DIR)
    # writer.flush()
