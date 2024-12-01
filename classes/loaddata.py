import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class OurDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations) # 4000 images
    
    def __getitem__(self, index):
        # row i, column 0 = name of the image
        # row i, column 1 = classificatin
        img_path = f"{self.root_dir}" +  f"{self.annotations.iloc[index, 0]}"
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        # optional transformation
        if self.transform:
            image = self.transform(image)

        return(image, y_label)