import os
import torch
import json 
import numpy 
import pandas as pd 

from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


class VehiclesDetectionDataset(Dataset): 
    def __init__(self, 
                 image_dir, 
                 label_path, 
                 transfoms=None, 
                 target_size= (640, 640)):
        self.image_dir= image_dir
        self.label_path= label_path
        self.target_size= target_size
        
        
        with open(label_path) as f:
            self.annotations = json.load(f)
        print(f"Initial annotations: {len(self.annotations)}")
        
        self.df_images = pd.DataFrame(self.annotations['images']).rename(columns={'id': 'image_id'})
        self.df_label= pd.DataFrame(self.annotations['annotations'])
        self.df = pd.merge(self.df_images, self.df_label, on='image_id', how='inner')
        
        # self.df['file_name']= self.image_dir + self.df['file_name']
        self.image_files = self.df["file_name"].unique()
        
        self.CLASSES= [cate['name'] for cate in self.annotations['categories']]
        self.CLASSES.remove('cars')
        
        self.LABEL_MAP = {}
        for cate in self.annotations['categories']: 
            self.LABEL_MAP[cate['name']] = cate['id']
        self.LABEL_MAP.pop('cars')
    
        self.transforms = transfoms
        if self.transforms is None:
            self.transforms = T.Compose([
                T.Resize(self.target_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            
    def __len__(self):
        return len(self.image_files)  
    
    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        df = pd.DataFrame(self.df).drop(['area', 'segmentation', 'iscrowd'], axis= 1) 
    
        bboxs = df['bbox'].values
        df['x_min'] = [int(bbox[0]) for bbox in bboxs]
        df['y_min'] = [int(bbox[1]) for bbox in bboxs]
        df['box_width'] = [int(bbox[2]) for bbox in bboxs]
        df['box_height'] = [int(bbox[3]) for bbox in bboxs]
    
        records = df[df["file_name"] == img_name]
        boxes = records[["x_min", "y_min", "box_width", "box_height"]].values
        labels = records["category_id"].values
        
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # Chuyển dữ liệu thành tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Tạo dictionary target
        target = {"boxes": boxes, "labels": labels}

        # Apply augmentations (nếu cần)
        if self.transforms:
            image = self.transforms(image)

        return image, target
        