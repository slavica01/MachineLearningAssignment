import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the CustomDataset class for Test Data
class TestDataset(Dataset):
    def __init__(self, csv_file, transform=None, base_dir="test_images"):
        """
        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
            base_dir (str): Base directory where images are stored.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        img_path = os.path.join(self.base_dir, img_path.lstrip('/'))

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} does not exist.")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        image_id = self.data.iloc[idx]['id']
        return image, image_id

def imshow(img, title=None):
    img = img / 2 + 0.5  # Unnormalize if normalized
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()

def main():
    # 2. Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Define CSV path for test data
    test_csv_path = "test_images_path.csv"

    # 4. Define the transformation pipeline (must match training transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model's expected input
        transforms.ToTensor(),          # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize as per ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

    # 5. Create Dataset and DataLoader instances for test data
    test_dataset = TestDataset(
        csv_file=test_csv_path,
        transform=transform,
        base_dir="test_images"  # Ensure this is the correct directory for test images
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,          # Adjust batch size as needed
        shuffle=False,
        num_workers=0,         # Set to 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )

    # 6. Load the trained model
    num_classes = 200  # Replace with the actual number of classes
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer
    model.to(device)

    # 7. Load the trained weights safely
    try:
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    except TypeError:
        # For older versions of PyTorch that do not support weights_only
        model.load_state_dict(torch.load("best_model.pth"))
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    model.eval()  # Set model to evaluation mode

    # 8. Initialize lists to store predictions and corresponding IDs
    predictions = []
    ids = []

    # 9. Perform inference on test data
    with torch.no_grad():
        for images, image_id in test_dataloader:
            images = images.to(device)  # Move images to device

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Store predictions and corresponding IDs
            predictions.extend(predicted.cpu().numpy() + 1)  # Adjust label indexing if necessary
            ids.extend(image_id.numpy())

    # 10. Create a DataFrame for predictions
    predictions_df = pd.DataFrame({'id': ids, 'label': predictions})

    # 11. Save predictions to a CSV file
    predictions_df.to_csv('test_predictions.csv', index=False)
    print("Test predictions have been saved to 'test_predictions.csv'.")

    # 12. Optional: Visualize a sample test image with its predicted label
    if len(test_dataset) > 0:
        sample_image, sample_id = test_dataset[0]
        model.eval()
        with torch.no_grad():
            image = sample_image.unsqueeze(0).to(device)  # Add batch dimension
            output = model(image)
            _, predicted_label = torch.max(output, 1)
            predicted_label = predicted_label.item() + 1  # Adjust if necessary

        imshow(sample_image, title=f"Image ID: {sample_id}, Predicted Label: {predicted_label}")
    else:
        print("Test dataset is empty. No images to visualize.")

if __name__ == '__main__':
    main()
