# config.py
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")

DATA_TRAIN_DIR = "C:/cache/torchok/data/sop/Stanford_Online_Products/Ebay_train.txt"
DATA_IMG_DIR = "C:/cache/torchok/data/sop/Stanford_Online_Products"

CHECKPOINT_PATH_SAVE_DIR = "C:Usets/Alexei/PycharmProjects/Search_Products/train/checkpoint/saved_model.pth"