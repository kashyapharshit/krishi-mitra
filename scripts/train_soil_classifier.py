import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 1. Model Definition
class CustomCNN(nn.Module):
    """A custom Convolutional Neural Network for image classification."""
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Flattened size calculation: 224 -> 112 -> 56 -> 28. So, 64x28x28
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Data Loading
def get_data_loaders(data_dir, batch_size):
    """Creates training and testing data loaders."""
    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Training or testing directory not found in '{data_dir}'. "
            "Please ensure the structure is data_dir/Train and data_dir/test"
        )

    # Note: Using ImageNet stats for normalization as in original script
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_dataset.classes

# 3. Training Loop
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    """Handles the model training loop."""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# 4. Evaluation
def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    return accuracy

# 5. Main Execution Block
def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description="Train a Custom CNN for Soil Classification.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the root data directory containing Train and test folders.')
    parser.add_argument('--model_output_path', type=str, default='models/soil_classifier.pth', help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    try:
        train_loader, test_loader, class_names = get_data_loaders(args.data_dir, args.batch_size)
        print(f"Found {len(class_names)} classes: {class_names}")
    except FileNotFoundError as e:
        print(e)
        return

    # Initialize model, loss, and optimizer
    num_classes = len(class_names)
    model = CustomCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    print("\nStarting training...")
    train_model(model, train_loader, criterion, optimizer, device, args.epochs)

    # Evaluate the model
    evaluate_model(model, test_loader, device)

    # Save the model
    os.makedirs(os.path.dirname(args.model_output_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_output_path)
    print(f"\nModel saved to {args.model_output_path}")

if __name__ == '__main__':
    main()
