import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from joblib import load as joblib_load
from PIL import Image
from torchvision import transforms

# Define the model architectures here to be able to load the state dicts
# This is a bit redundant, but necessary without a more complex module system.

class CustomCNN(nn.Module):
    """A custom Convolutional Neural Network for image classification."""
    def __init__(self, num_classes=4): # Defaulting to 4 classes as per original data
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CropNN(nn.Module):
    """Feed-forward Neural Network for crop recommendation."""
    def __init__(self, input_size=15, num_classes=22): # Defaults based on original data
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


def load_artifacts(models_dir):
    """Loads all necessary models and preprocessing objects."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Soil Classifier
    soil_model = CustomCNN()
    soil_model.load_state_dict(torch.load(os.path.join(models_dir, 'soil_classifier.pth'), map_location=device))
    soil_model.to(device).eval()

    # Load Crop Recommender
    crop_model = CropNN()
    crop_model.load_state_dict(torch.load(os.path.join(models_dir, 'crop_recommender.pth'), map_location=device))
    crop_model.to(device).eval()

    # Load Scaler and Label Encoder
    scaler = joblib_load(os.path.join(models_dir, 'scaler.joblib'))
    label_encoder = joblib_load(os.path.join(models_dir, 'label_encoder.joblib'))

    return soil_model, crop_model, scaler, label_encoder, device

def predict_soil(image_path, model, device):
    """Predicts soil type from an image file."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # The class names are inferred from the folder names during training.
    # We are hardcoding them here based on the original project's dataset.
    # A better solution would be to save class_names with the model.
    class_names = ['Alluvial', 'Black', 'Clay', 'Red']

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class.lower()

def main():
    """Main application function."""
    parser = argparse.ArgumentParser(description="Krishi Mitra: AI-Based Agriculture Assistant")
    parser.add_argument('--models_dir', type=str, default='models/', help='Directory where trained models and artifacts are stored.')
    args = parser.parse_args()

    try:
        soil_model, crop_model, scaler, le, device = load_artifacts(args.models_dir)
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        print("Please ensure you have trained the models using the scripts in the `scripts/` directory first.")
        return

    print("🌾 Welcome to Krishi Mitra – Your AI Agriculture Assistant 🌾")

    # --- Get Soil Type ---
    soil_type = ''
    soil_map = {'alluvial': 0, 'black': 1, 'red': 2, 'laterite': 3}

    while True:
        user_knows_soil = input("🧭 Do you know your soil type? (yes/no): ").strip().lower()
        if user_knows_soil == "yes":
            soil_input = input(f"🟤 Enter soil type ({' / '.join(soil_map.keys())}): ").strip().lower()
            if soil_input in soil_map:
                soil_type = soil_input
                break
            else:
                print("❌ Invalid soil type. Please choose from the list.")
        elif user_knows_soil == "no":
            image_path = input("📁 Enter the full path to your soil image: ").strip()
            if os.path.isfile(image_path):
                try:
                    soil_type = predict_soil(image_path, soil_model, device)
                    print(f"\n🧠 Predicted Soil Type: {soil_type.capitalize()}")
                    break
                except Exception as e:
                    print(f"❌ Error predicting from image: {e}")
            else:
                print("❌ Invalid file path. Please try again.")
        else:
            print("⚠️ Please answer 'yes' or 'no'.")

    # --- Get NPK values ---
    npk_defaults = {'alluvial': (90, 35, 45), 'black': (100, 40, 50), 'red': (50, 25, 70), 'laterite': (40, 22, 60)}
    while True:
        use_auto_npk = input("\n🔄 Auto-fill N, P, K values based on soil type? (yes/no): ").strip().lower()
        if use_auto_npk == 'yes':
            N, P, K = npk_defaults.get(soil_type, (70, 30, 60)) # Fallback default
            print(f"   🧪 Nitrogen (N): {N}, Phosphorus (P): {P}, Potassium (K): {K}")
            break
        elif use_auto_npk == 'no':
            try:
                N = float(input("   Enter Nitrogen (N): "))
                P = float(input("   Enter Phosphorus (P): "))
                K = float(input("   Enter Potassium (K): "))
                break
            except ValueError:
                print("❌ Invalid input. Please enter numbers.")
        else:
            print("⚠️ Please answer 'yes' or 'no'.")

    # --- Get Environmental Factors ---
    while True:
        try:
            temp = float(input("\n🌡️ Enter Temperature (°C): "))
            humidity = float(input("💧 Enter Humidity (%): "))
            ph = float(input("🧪 Enter soil pH: "))
            rainfall = float(input("🌧️ Enter Rainfall (mm): "))
            break
        except ValueError:
            print("❌ Invalid input. Please enter numbers for all fields.")

    # --- Prepare data for Crop Model ---
    # The number of soil classes for one-hot encoding is assumed to be 8 from the original script
    # This is a discrepancy in the original project (4 image classes vs 8 one-hot classes)
    # We will follow the crop recommender's expectation of 8 classes.
    # The known soil types will be mapped, others will not be one-hot encoded.
    soil_one_hot = np.zeros(8)
    if soil_type in soil_map:
        soil_one_hot[soil_map[soil_type]] = 1

    numerical_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    scaled_data = scaler.transform(numerical_data)

    final_input = np.concatenate([scaled_data, soil_one_hot.reshape(1, -1)], axis=1).astype(np.float32)
    input_tensor = torch.tensor(final_input, dtype=torch.float32).to(device)

    # --- Predict Crop ---
    with torch.no_grad():
        output = crop_model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    top_indices = probs.argsort()[::-1][:3]
    top_crops = le.inverse_transform(top_indices)
    top_probs = probs[top_indices]

    print("\n" + "="*30)
    print("📈 TOP 3 CROP RECOMMENDATIONS")
    print("="*30)
    for i in range(3):
        print(f"  {i+1}. {top_crops[i]:<15} (Confidence: {top_probs[i]:.2%})")
    print("-" * 30)
    print(f"✅ Best Suited Crop: {top_crops[0]}")
    print("="*30)


if __name__ == '__main__':
    main()
