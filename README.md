# 🌾 Krishi Mitra – AI-Based Agriculture Assistant

**Krishi Mitra** is an AI-powered smart agriculture system to assist Indian farmers in making data-driven decisions regarding crop selection. It uses a combination of a Convolutional Neural Network (CNN) for soil image classification and a Feed-forward Neural Network (FNN) for crop recommendation based on environmental factors.

This project was originally developed during a short-term internship at **CDAC Patna** and has been refactored for clarity, maintainability, and ease of use.

---

## 📂 Project Structure

The repository is now organized into a clean and professional structure:

```
krishi-mitra/
│
├── data/
│   ├── (Your raw data should be placed here)
│   └── augmented_crops.csv         # Auto-generated during training
│
├── models/
│   ├── soil_classifier.pth         # Trained CNN model
│   ├── crop_recommender.pth        # Trained FNN model
│   ├── scaler.joblib               # StandardScaler for features
│   └── label_encoder.joblib        # LabelEncoder for crop names
│
├── scripts/
│   ├── train_soil_classifier.py    # Script to train the soil image model
│   └── train_crop_recommender.py   # Script to train the crop recommendation model
│
├── src/
│   └── app.py                      # Main user-facing application
│
├── .gitignore
├── README.md                       # This documentation
└── requirements.txt                # Python dependencies
```

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Setup

First, clone the repository to your local machine:
```bash
git clone <repository_url>
cd krishi-mitra
```

Next, install the required Python dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Datasets

This project requires two datasets: one for soil image classification and one for crop recommendation. The original links are provided below.

- **Soil Image Dataset:** [Access on Google Drive](https://drive.google.com/drive/folders/1K3FY6TzLudNfStd32DtfBby-IkpAoYBo?usp=sharing)
- **Crop Data (new_agri.csv):** [Access on Google Drive](https://drive.google.com/file/d/1dDUNgzlXCpM7KiR6JkwPL1mZ7QU1eTCz/view?usp=sharing)

**IMPORTANT:**
1.  Download the **Soil Image Dataset**. Create a directory inside the `data/` folder (e.g., `data/soil_images`) and ensure it contains the `Train` and `test` subdirectories from the dataset.
2.  Download the **Crop Data** and place the `new_agri.csv` file directly inside the `data/` folder.

### 3. Usage

The project workflow is split into two main parts: training the models and running the application.

#### Training the Models

You must train the models first. The trained models and artifacts will be saved in the `models/` directory.

**A. Train the Soil Classifier:**
Run the following command, pointing to the directory where you stored the soil images:
```bash
python scripts/train_soil_classifier.py --data_dir data/soil_images/
```

**B. Train the Crop Recommender:**
Run the following command, pointing to the raw `new_agri.csv` file:
```bash
python scripts/train_crop_recommender.py --raw_data_path data/new_agri.csv
```

#### Running the Application

Once the models are trained and saved, you can run the main application:
```bash
python src/app.py
```
The application will guide you through a series of prompts to get a personalized crop recommendation.

---

### 🧰 Original Technologies Used
- Python
- PyTorch
- NumPy, Pandas, Matplotlib
- Scikit-learn

### 🙏 Acknowledgement
This project was developed during a short-term AI internship at CDAC Patna.
> **Original Team:** Harshit Kashyap, Ayush Kumar, Pallavi Bharti, Sonu Kumar
