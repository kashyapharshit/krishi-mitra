# 🌾 Krishi Mitra – AI-Based Agriculture Assistant

**Krishi Mitra** is an AI-powered smart agriculture system developed during a short-term internship at **CDAC Patna** (June 2025). It aims to assist Indian farmers in making data-driven decisions regarding crop selection, irrigation, and market timing using machine learning and deep learning techniques.

This repository contains the **core model training code**, including:
- Crop recommendation using tabular soil data
- Soil classification using image data
- Price forecasting (LSTM support)

  
📥 for  Dataset Access go below.


---

## 📌 Problem Statement

Indian farmers face challenges such as:
- Unscientific crop selection
- Poor awareness of soil conditions
- Water mismanagement
- Lack of real-time market price insights

**Krishi Mitra** solves these problems using AI by providing personalized suggestions through crop modeling, soil analysis, and predictive forecasting.

---

## 🧠 Project Features

- **Crop Recommendation Model**  
  Predicts the best crop to grow based on pH, soil type, and location using a PyTorch Feedforward Neural Network.

- **Soil Classification Model**  
  A custom Convolutional Neural Network (CNN) classifies soil types using images when sensor data is missing.

- **Market Price Forecasting (Support)**  
  Time-series prediction using LSTM to estimate mandi prices (future extension).
  
#MODEL ARCHITECTURE
krishi-mitra/
│
├── model/
│   ├── crop_recommendation_model.py     # PyTorch FNN for crop prediction
│   └── soil_image_classifier.py         # CNN model for soil image classification
│
├── data/                                # Contains crop dataset and image folders
│
├── requirements.txt                     # Python dependencies
├── LICENSE                              # MIT License
├── README.md                            # This documentation
└── .gitignore                           # Git tracking exclusions


🧰 Technologies Used

Python

PyTorch, TensorFlow, Keras

NumPy, Pandas, Matplotlib

OpenWeatherMap API (future scope)

Agmarknet API (market price support)



---

🙏 Acknowledgement

This project was developed during a short-term AI internship at CDAC Patna under the guidance of mentors and in collaboration with team members.

> Team: Harshit Kashyap, Ayush Kumar, Pallavi Bharti, Sonu Kumar

## 📥 Dataset Access 

The datasets used in this project (soil images) are available on Google Drive.

[Click here to access the dataset](https://drive.google.com/drive/folders/1K3FY6TzLudNfStd32DtfBby-IkpAoYBo?usp=sharing)

## 📥 Dataset Access

The datasets used in this project (crop data ) are available on Google Drive.

[Click here to access the dataset](https://drive.google.com/file/d/1dDUNgzlXCpM7KiR6JkwPL1mZ7QU1eTCz/view?usp=sharing)

Make sure to download and place the files inside the `data/` folder before running the models.

Make sure to download and place the files inside the `data/` folder before running the models.
