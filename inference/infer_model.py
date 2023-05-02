import torch
import torch.nn as nn

#internal imports
from train.model import MyModel
from config import config

def load_model():
    model = MyModel()
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH_LOAD_DIR, map_location ='cpu'))
    submodules = list(model.children())
    # удаляем стандартный классификатор
    model = nn.Sequential(*submodules[:-1])
    model.to(config.DEVICE)
    model.eval()

load_model()