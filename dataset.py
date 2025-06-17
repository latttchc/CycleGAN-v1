import os 
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_trainA, root_trainB, transform=None):
        self.root_trainA = root_trainA
        self.root_trainB = root_trainB
        self.transform = transform

        self.trainA_images = os.listdir(root_trainA)
        self.trainB_images = os.listdir(root_trainB)

        self.length_dataset  = max(len(self.zebra_images), len(self.horse_images))
        self.trainA_len = len(self.trainA_images)
        self.trainB_len = len(self.trainB_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        trainA_img = self.trainA_images[index % self.trainA_len]
        trainB_img = self.trainB_images[index % self.trainB_len]

        trainA_path = os.path.join(self.root_trainA, trainA_img)
        trainB_path = os.path.join(self.root_trainB, trainB_img)

        trainA_image = Image.open(trainA_path).convert("RGB")
        trainB_image = Image.open(trainB_path).convert("RGB")

        if self.transform:
            transformed = self.transform(image=trainA_image, image0=trainB_image)
            trainA_image = transformed["image"]
            trainB_image = transformed["image0"]

        return trainA_image, trainB_image