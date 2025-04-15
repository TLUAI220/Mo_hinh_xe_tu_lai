import os 
import yaml
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from lib.data.load import LoadData
from lib.data.loader import VehiclesDetectionDataset
from lib.model.train import train_and_evaluate
from lib.model.models.__dtc_obj import FasterRCNN
# from lib.model.test import valid
from torch.utils.data import DataLoader


import torchvision.models.detection as detection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch.optim as optim

device= "cuda" if torch.cuda.is_available() else "cpu"

#|----------------------------------| 
# LOAD CONFIG
#|----------------------------------| 
with open("cfg.yaml", "r", encoding="utf-8") as file: 
    config = yaml.safe_load(file) 
print(config)

#|----------------------------------| 
# LOAD DATA
#|----------------------------------| 

LOAD_DATA_CFG= config['DATASET']['LOAD_DATA']
FILE_ID= LOAD_DATA_CFG['FILE_ID']
FILE_ZIP= LOAD_DATA_CFG['FILE_ZIP']
EXTRACT_FOLDER= LOAD_DATA_CFG['EXTRACT_FOLDER']

LOAD_DATA= LoadData(file_id= FILE_ID, 
                    file_zip= FILE_ZIP, 
                    extract_folder= EXTRACT_FOLDER)
ROOT= LOAD_DATA.root
print(f"FOLDER: {ROOT}")

# LOAD_DATA.__download__()

#|----------------------------------| 
# LOAD DATA
#|----------------------------------| 

IMAGE_DIR= f"{ROOT}/data/images/Vehicles_Detection.v9i.coco/train"
LABEL_PATH= f"{ROOT}/data/labels/Vehicles_Detection.v9i.coco/train_annotations.coco.json"
train_set= VehiclesDetectionDataset(image_dir= IMAGE_DIR, 
                                    label_path= LABEL_PATH)

IMAGE_DIR= f"{ROOT}/data/images/Vehicles_Detection.v9i.coco/valid"
LABEL_PATH= f"{ROOT}/data/labels/Vehicles_Detection.v9i.coco/valid_annotations.coco.json"
valid_set= VehiclesDetectionDataset(image_dir= IMAGE_DIR, 
                                    label_path= LABEL_PATH)
NUM_CLASSES = len(train_set.CLASSES)


LOADER = config['DATASET']['LOADER']
BATCH_SIZE= LOADER['BATCH_SIZE']
SHUFFE= LOADER['SHUFFLE']
# NUM_WORKER= LOADER['NUM_WORKERS']

train_loader= DataLoader(train_set, 
                         batch_size= BATCH_SIZE, 
                         shuffle= SHUFFE, 
                         collate_fn=lambda batch: tuple(zip(*batch)))
valid_loader= DataLoader(valid_set, 
                         batch_size= BATCH_SIZE, 
                         shuffle= SHUFFE, 
                         collate_fn=lambda batch: tuple(zip(*batch)))

# # #|----------------------------------| 
# # # MODEL
# # #|----------------------------------| 

num_classes = NUM_CLASSES + 1   # 1 class + background
model = FasterRCNN(num_classes)
model.to(device)

#|----------------------------------| 
# TRAIN
#|----------------------------------| 
# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)    
model_trained = train_and_evaluate(model, train_loader, valid_loader, optimizer, num_epochs=5)
torch.save(model_trained.state_dict(), f'{ROOT}/save/Faster_RCNN.pth')


