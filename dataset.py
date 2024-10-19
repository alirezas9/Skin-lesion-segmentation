import os
import cv2 as cv
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class data(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        img_path = os.path.join(self.image_dir, name) 
        mask_path = os.path.join(self.mask_dir, name.replace(".jpg", "_Segmentation.png"))
        
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")
        # for augmentation
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # image = np.array(image)
        # mask = np.array(, dtype=np.float32)
        # mask[mask == 255.0] = 1.0
        mask = mask/255.0

            

        return image, mask
        