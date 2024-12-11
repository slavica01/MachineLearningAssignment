import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# 1. Define the CustomDataset class at the top level
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, base_dir="train_images", has_labels=True):
        """
        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
            base_dir (str): Base directory where images are stored.
            has_labels (bool): Indicates if the dataset includes labels.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.base_dir = base_dir
        self.has_labels = has_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")

        img_path = self.data.iloc[idx]['image_path']
        img_path = os.path.join(self.base_dir, img_path.lstrip('/'))

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} does not exist.")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label = self.data.iloc[idx]['label']
            label = torch.tensor(label - 1, dtype=torch.long)  # Adjust if labels start at 1
            return image, label
        else:
            image_id = self.data.iloc[idx]['id']
            return image, image_id

def main():
    # 2. Check if MPS backend is available
    if torch.backends.mps.is_available():
        print("MPS backend is available. Running on Apple GPU.")
    else:
        print("MPS backend is not available. Running on CPU.")

    # 3. Set device (MPS if available, otherwise CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 4. Define CSV paths (relative to current directory)
    train_csv_path = "train_images.csv"
    test_csv_path = "test_images_path.csv"

    # 5. Read the CSV files
    train_label_df = pd.read_csv(train_csv_path)
    test_label_df = pd.read_csv(test_csv_path)

    # 6. Retain the 'id' column in the test CSV for mapping predictions
    #    If 'label' exists and is not needed, drop it
    if 'label' in test_label_df.columns:
        test_label_df = test_label_df.drop(columns=['label'])

    # 7. Split training data into training and validation sets
    train_df, val_df = train_test_split(
        train_label_df,
        test_size=0.2,  # 20% for validation
        stratify=train_label_df['label'],  # Maintain label distribution
        random_state=42  # For reproducibility
    )

    # 8. Save the splits to new CSV files (optional)
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)

    # 9. Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images for ResNet18
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize to ImageNet standards
                             std=[0.229, 0.224, 0.225])
    ])

    # 10. Create Dataset instances
    train_dataset = CustomDataset(csv_file='train_split.csv', transform=transform, base_dir="train_images", has_labels=True)
    val_dataset = CustomDataset(csv_file='val_split.csv', transform=transform, base_dir="train_images", has_labels=True)
    test_dataset = CustomDataset(csv_file=test_csv_path, transform=transform, base_dir="train_images", has_labels=False)

    # 11. Create DataLoader instances
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)  # Test in batch_size=1 for evaluation

    # 12. Define hyperparameters
    learning_rate = 0.0001
    num_epochs = 10
    patience = 5  # Stop training after 5 epochs with no improvement
    patience_counter = 0

    # 13. Compute class weights for imbalanced dataset
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_label_df['label']),
        y=train_label_df['label']
    )

    # 14. Convert class weights to tensor and move it to the device
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 15. Load ResNet18 pretrained model with updated syntax to avoid warnings
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Updated syntax

    # 16. Modify the final fully connected layer to match the number of classes in your dataset
    num_classes = len(train_label_df['label'].unique())  # Automatically calculate the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the last FC layer to match your dataset's class count

    # 17. Send model to device (MPS/CPU)
    model.to(device)

    # 18. Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use class weights in loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 19. Initialize Early Stopping Variables
    best_val_loss = float('inf')  # Initialize to infinity
    patience_counter = 0         # Reset counter when improvement occurs

    # 20. Training loop with validation
    for epoch in range(num_epochs):
        # Training Phase
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to device

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_dataloader)
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_dataloader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)  # Move data to device

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_dataloader)
        val_epoch_accuracy = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%")

        # Early stopping check
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            # Save the best model checkpoint
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            print(f"  -> No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping after epoch {epoch+1}")
                break

    # 21. Load the best model for testing
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()  # Set the model to evaluation mode

    # 22. Initialize lists to store predictions and corresponding IDs
    predictions = []
    ids = []

    with torch.no_grad():
        for images, image_id in test_dataloader:
            images = images.to(device)  # Move images to device

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Store predictions and corresponding IDs
            predictions.append(predicted.cpu().numpy()[0] + 1)  # Adjust label indexing if necessary
            ids.append(image_id.numpy()[0])

    # 23. Create a DataFrame for predictions
    predictions_df = pd.DataFrame({'id': ids, 'label': predictions})

    # 24. Save predictions to a CSV file
    predictions_df.to_csv('test_predictions.csv', index=False)
    print("Test predictions have been saved to 'test_predictions.csv'.")

    # 25. Optional: Visualize some test images with their predicted labels
    def imshow(img, title=None):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if title:
            plt.title(title)
        plt.show()

    # Display the first test image with its predicted label
    dataiter = iter(test_dataloader)
    images, image_id = next(dataiter)
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)
    imshow(images[0], title=f"Predicted Label: {predicted.item() + 1}")

if __name__ == '__main__':
    main()
