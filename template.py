# import dataset
from classes.loaddata import OurDataset
# import np
import numpy as np
# import torch
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
# import pickle
import pickle
# image processer
from PIL import Image

class CNN(nn.Module):
    def __init__(self, in_channel = 3, num_classes = 200):
        super(CNN, self).__init__()
        """
        Layers here
        """
        
    def forward(self, x):
        """
        Forward function here
        """
        return x
    
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    # dont use gradient to check cause we dont need it here
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            y = y - 1
            scores = model(x)
            # find index for second dimension?
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)
        
        print(f"got {num_correct} / {num_samples} with accuracy {(float(num_correct)/float(num_samples))*100:.2f}")
    
    model.train()
    accuracy = (float(num_correct)/float(num_samples))*100
    return accuracy

def save_checpoint(state, filename):
    print("saving checkpoint")
    torch.save(state, filename)

def load_checpoint(checkpoint):
    print("loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

my_transforms = transforms.Compose([
    transforms.Lambda(lambda img: Image.fromarray(img) if isinstance(img, np.ndarray) else img),  
    transforms.Resize((TODO, TODO)),  
    transforms.Lambda(lambda img: img.convert('RGB')),  
    transforms.ToTensor(),
    """
    Image transformations here. 
    Note: adjust transforms.resize
    Note: The others above are necessary
    """
])

# Hyperparameters
input_size = 32
num_classes = 200
learning_rate = 0.001
batch_size = 32
num_epochs = 50

# initialize dataset
"""
Takes a 90% training and 10% test dataset
"""
dataset = OurDataset(csv_file = "datafile/train_images.csv", root_dir = "datafile/train_images", transform = my_transforms)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [3533, 393])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

class_dictionary = np.load("datafile/class_names.npy", allow_pickle=True).item()
class_names = list(class_dictionary.keys())
class_names = [name.split('.',1)[1] for name in class_names]

# NOTE: If set to False you will overwrite previous results
# NOTE: In first run needs to be set to false
load_model = True # NOTE: If set to False you will overwrite previous results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

###################################################
"""
Test code to check if there are no bugs in code
"""
# data, targets = next(iter(train_loader))
####################################################

"""
Define the file that you save your model to here

Note: please read notes at initalizing load_model variable
"""
filename = TODO
if load_model:
    load_checpoint(torch.load(f"{filename}.pth.tar"))

"""
Main run below
"""

# initialize lists to track loss and accuracy
train_losses = []
test_accuracies = []

# Train network
for epoch in range(num_epochs):
    print(f"epoch nr. {epoch}")

    if epoch % 5 == 0:
        # Save state
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checpoint(checkpoint)
    
    epoch_loss = 0
    # data: image, target: label for each image
    for batch_index, (data, targets) in enumerate(train_loader):
        # to cuda if possible else cpu
        data = data.to(device=device)
        targets=targets.to(device=device)

        # print(data.shape)

        # forward
        scores = model(data)
        loss = loss_function(scores, targets - 1)
       
        # might use this for plots
        # train_losses.append(loss.item())

        # backward
        # set to zero from previous forward props
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

        epoch_loss += loss.item()

    # check accuracy

    # print("checking train set....")
    # accuracy = check_accuracy(train_loader, model)
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Train Loss: {avg_epoch_loss:.4f}")
    print("checking test set...")
    accuracy = check_accuracy(test_loader, model)
    print(f"Epoch {epoch}, Test Accuracy: {accuracy:.2f}%")