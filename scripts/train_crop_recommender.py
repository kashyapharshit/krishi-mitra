import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

# --- Data Augmentation ---
def augment_data(raw_data_path, output_path):
    """
    Loads a raw crop dataset, generates synthetic data based on predefined
    crop settings, and saves the augmented dataset.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_data_path}")

    df = pd.read_csv(raw_data_path)

    # These settings were in the original script for augmenting the dataset
    crop_settings = {
        'wheat': {'temp': (18, 25), 'rain': (500, 750), 'humid': (60, 85), 'ph': (6.0, 7.5), 'N': (100, 150), 'P': (40, 60), 'K': (40, 60), 'soil_preference': [0, 3]},
        'sugarcane': {'temp': (25, 35), 'rain': (1000, 1500), 'humid': (70, 90), 'ph': (6.0, 7.5), 'N': (150, 250), 'P': (50, 80), 'K': (80, 120), 'soil_preference': [0, 3]},
        'potato': {'temp': (18, 25), 'rain': (500, 700), 'humid': (65, 85), 'ph': (5.0, 6.0), 'N': (100, 150), 'P': (60, 90), 'K': (120, 180), 'soil_preference': [0, 1, 3]},
        'litchi': {'temp': (24, 32), 'rain': (1200, 1500), 'humid': (75, 95), 'ph': (5.5, 6.5), 'N': (70, 120), 'P': (30, 50), 'K': (50, 80), 'soil_preference': [0, 3]},
        'makhana': {'temp': (25, 35), 'rain': (1200, 2000), 'humid': (80, 100), 'ph': (6.0, 7.5), 'N': (50, 100), 'P': (25, 45), 'K': (35, 60), 'soil_preference': [0, 2]},
        'tomato': {'temp': (20, 30), 'rain': (600, 900), 'humid': (60, 85), 'ph': (6.0, 7.0), 'N': (80, 150), 'P': (50, 90), 'K': (80, 150), 'soil_preference': [0, 3]},
        'onion': {'temp': (20, 30), 'rain': (600, 800), 'humid': (60, 80), 'ph': (6.0, 7.0), 'N': (80, 120), 'P': (40, 60), 'K': (80, 100), 'soil_preference': [0, 1, 3]}
    }

    augmented = []
    for crop, params in crop_settings.items():
        for _ in range(100):
            soil_type = np.random.choice(params['soil_preference'])
            soil_vector = [1 if i == soil_type else 0 for i in range(4)]
            augmented.append([
                np.random.uniform(*params['N']), np.random.uniform(*params['P']), np.random.uniform(*params['K']),
                np.random.uniform(*params['temp']), np.random.uniform(*params['humid']), np.random.uniform(*params['ph']),
                np.random.uniform(*params['rain']), crop, *soil_vector
            ])

    aug_df = pd.DataFrame(augmented, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label', 'soil_0', 'soil_1', 'soil_2', 'soil_3'])

    # Ensure original dataframe has the soil columns to allow concatenation
    for col in ['soil_0', 'soil_1', 'soil_2', 'soil_3']:
        if col not in df.columns:
            df[col] = 0

    df_aug = pd.concat([df, aug_df], ignore_index=True)
    df_aug.to_csv(output_path, index=False)
    print(f"Augmented data saved to {output_path} with shape {df_aug.shape}")
    return output_path

# --- Data Preparation ---
def prepare_data(augmented_csv_path):
    """Loads and preprocesses the augmented data for training."""
    df = pd.read_csv(augmented_csv_path)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    # Assuming first 7 cols are numeric, rest are one-hot encoded soil types
    X = df.drop('label', axis=1).values
    y = df['label'].values

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[:, :7])
    X_cat = X[:, 7:]
    X_final = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, le

# --- PyTorch Dataset ---
class CropDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Model Definition ---
class CropNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CropNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.out(x)

# --- Training Loop ---
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, patience):
    """Handles the model training loop with early stopping."""
    best_val_acc = 0
    counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        # Training phase
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        print(f"Epoch [{epoch+1}/{epochs}], Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered. Best Val Acc: {best_val_acc:.2f}%")
                break

    model.load_state_dict(best_model_state)
    return model

# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Train a Crop Recommendation NN.")
    parser.add_argument('--raw_data_path', type=str, required=True, help='Path to the raw crop data CSV file (e.g., new_agri.csv).')
    parser.add_argument('--output_dir', type=str, default='models/', help='Directory to save the trained model and artifacts.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Augment data
    augmented_data_path = os.path.join(os.path.dirname(args.output_dir), 'data', 'augmented_crops.csv')
    os.makedirs(os.path.dirname(augmented_data_path), exist_ok=True)
    augment_data(args.raw_data_path, augmented_data_path)

    # 2. Prepare data
    X_train, X_test, y_train, y_test, scaler, le = prepare_data(augmented_data_path)
    train_dataset = CropDataset(X_train, y_train)
    test_dataset = CropDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 3. Initialize and train model
    input_size = X_train.shape[1]
    num_classes = len(le.classes_)
    model = CropNN(input_size=input_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("\nStarting training...")
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, args.epochs, args.patience)

    # 4. Final evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n--- Final Evaluation ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # 5. Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'crop_recommender.pth'))
    dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))
    dump(le, os.path.join(args.output_dir, 'label_encoder.joblib'))
    print(f"\nModel and artifacts saved to {args.output_dir}")

if __name__ == '__main__':
    main()
