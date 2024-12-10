import os
import torch
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from CNN_train import CNNWithAttributes
from PIL import Image

class OurTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, has_id_column=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.has_id_column = has_id_column

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        try:
            # Get image path and image ID
            if self.has_id_column:
                img_name = str(self.annotations.iloc[index, 0])
                img_rel_path = self.annotations.iloc[index, 1].strip()
            else:
                img_name = 0  # Not used
                img_rel_path = self.annotations.iloc[index, 0].strip()

            # Fix img_rel_path and construct full path
            img_rel_path = img_rel_path.strip("/")
            img_path = os.path.join(self.root_dir, img_rel_path)

            # Check if file exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"File not found: {img_path}")

            # Load the image
            image = Image.open(img_path).convert("RGB")

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            return image, img_name

        except Exception as e:
            print(f"Error at index {index}: {e}")
            raise e
        
# Define testing function
def test_model(model, test_loader, device, attribute_dim, output_csv="test.csv"):
    model.eval()
    results = []

    # Create a placeholder tensor for attributes
    placeholder_attributes = torch.zeros((1, attribute_dim)).to(device)

    with torch.no_grad():
        for data, image_ids in test_loader:
            data = data.to(device)

            # Use the placeholder attributes
            attributes = placeholder_attributes.repeat(data.size(0), 1)

            # Get predictions
            outputs = model(data, attributes)
            _, predicted = outputs.max(1)

            # Collect predictions
            for img_id, pred in zip(image_ids, predicted.cpu().numpy()):
                results.append((img_id, pred))

    # Save predictions to CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "label"])
        writer.writerows(results)

    print(f"Test predictions saved to {output_csv}")


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
checkpoint_path = "checkpoints/checkpoint_epoch_20.pth.tar" 
checkpoint = torch.load(checkpoint_path)
model = CNNWithAttributes(in_channel=3, num_classes=200, attribute_dim=312).to(device)
model.load_state_dict(checkpoint['state_dict'])

# Set model to evaluation mode
model.eval()

# Define test dataset
test_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = OurTestDataset(
    csv_file="aml-2024-feather-in-focus/test_images_path.csv",
    root_dir="aml-2024-feather-in-focus/test_images/",
    transform=test_transforms,
    has_id_column=True
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Run the testing process
if __name__ == "__main__":
    test_model(model, test_loader, device, attribute_dim=312)

