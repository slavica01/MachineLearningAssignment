# import dataset

# import np
import numpy as np
import pandas as pd
# import torch
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from skimage import io
from torch.utils.data import Dataset
import pickle
import os
# image processer
from PIL import Image

class OurDataset(Dataset):
    def __init__(self, csv_file, root_dir, attributes, has_id_column=False, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.attributes = attributes
        self.has_id_column = has_id_column
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        try:
            # Get image path and label
            if self.has_id_column:
                img_rel_path = self.annotations.iloc[index, 1].strip()
                label = int(self.annotations.iloc[index, 2])
            else:
                img_rel_path = self.annotations.iloc[index, 0].strip()
                label = int(self.annotations.iloc[index, 1])

            # Fix img_rel_path and construct full path
            img_rel_path = img_rel_path.strip("/")
            img_path = os.path.join(self.root_dir, img_rel_path)

            # Check if file exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"File not found: {img_path}")

            # Load the image
            image = io.imread(img_path)

            # Convert to PIL image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            # Get the attribute vector for the class
            label = label - 1
            attr_vector = torch.tensor(self.attributes[label], dtype=torch.float)

            return image, attr_vector, label

        except Exception as e:
            print(f"Error at index {index}: {e}")
            raise e

    
# Load the attribute and class name files
attributes = np.load("aml-2024-feather-in-focus/attributes.npy", allow_pickle=True)
class_names = np.load("aml-2024-feather-in-focus/class_names.npy", allow_pickle=True)


# transform the training and testing dataset
train_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the train dataset
train_dataset = OurDataset(
    csv_file = "aml-2024-feather-in-focus/train_images.csv",
    root_dir = "aml-2024-feather-in-focus/train_images",
    attributes = attributes,
    has_id_column = False,
    transform = train_transforms
)

# Load the test dataset
test_dataset = OurDataset(
    csv_file = "aml-2024-feather-in-focus/test_images_path.csv",
    root_dir = "aml-2024-feather-in-focus/test_images/",
    attributes = attributes,
    has_id_column = True,
    transform = test_transforms
)

# create and validation set
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# create data loader
train_loader = DataLoader(train_subset, batch_size=64, shuffle = False)
val_loader = DataLoader(val_subset, batch_size=64, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle = False)

# define CNN with attributes class
class CNNWithAttributes(nn.Module):
    def __init__(self, in_channel=3, num_classes=201, attribute_dim=312):
        super(CNNWithAttributes, self).__init__()
        # First set of convolution layers
        self.conv1 = nn.Conv2d(in_channel, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Second set of convolution layers
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fully connected layers for CNN features
        self.fc1 = nn.Linear(16 * 8 * 8, out_features=500)
        self.relu3 = nn.ReLU()

        # Attribute feature extractor
        self.attribute_fc = nn.Linear(attribute_dim, 64)  

        # Combined features for classification
        self.fc2 = nn.Linear(500 + 64, num_classes) 

    def forward(self, x, attributes):
        # CNN feature extraction
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten CNN output and pass through fully connected layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # Attribute feature extraction
        attr_features = self.attribute_fc(attributes)
        attr_features = torch.relu(attr_features)

        # Combine CNN features and attribute features
        combined = torch.cat((x, attr_features), dim=1)

        # Final classification
        output = self.fc2(combined)

        return output


# train out model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data, att, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs = model(data,att)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

        # Validate the model
        validate_model(model, val_loader, criterion)

# define the function to validate the model   
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, att, targets in val_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data,att)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# define the function to test the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data,att, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data,att)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# define the device, model, criterion and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNWithAttributes(in_channel=3, num_classes = 200).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# now run the model'
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs = 15)
test_model(model, test_loader)
torch.save(model.state_dict(), "trained_model.pth")
model.eval()