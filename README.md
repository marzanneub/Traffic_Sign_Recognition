## 🚦 Traffic Sign Recognition using CNN & Streamlit

This project is a deep learning-based **Traffic Sign Recognition** system trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. It uses a Convolutional Neural Network (CNN) to classify images into 43 traffic sign categories and includes a **Streamlit web app** for real-time prediction from uploaded images.

### 📁 Project Structure

.

├── Train_Traffic_Sign_Recognition.ipynb # Google Colab notebook for training the model

├── Test_Traffic_Sign_Recognition.ipynb # Google Colab notebook for testing the model

├── main.py # Streamlit web app for sign prediction

├── trained_model.h5 # Saved trained CNN model

├── training_hist.json # Training metrics (accuracy/loss) in JSON

├── Train.zip # Raw training dataset (GTSRB format)

├── Test_Photos.zip # Raw testing dataset

├── README.md # Project documentation (you are here)

## 🧠 Features

- Built with TensorFlow and Keras  
- Uses CNNs for feature extraction and classification  
- Automatically loads zipped training/testing data from Google Drive  
- Training history saved to `training_hist.json`  
- Visualizes training and validation accuracy with Matplotlib  
- Web interface via Streamlit for uploading and predicting traffic signs  

## 🧪 Dataset Details

- Source: [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_dataset.html)  
- Classes: 43 traffic sign categories  
- Image Dimensions: 128×128 pixels  
- Training Images Used: 31,368  
- Validation Images Used: 7,841  

## 📊 Model Architecture

The model is a deep CNN with the following layers:

- Conv2D (32 filters) → Conv2D (32) → MaxPooling  
- Conv2D (64) → Conv2D (64) → MaxPooling  
- Conv2D (128) → Conv2D (128) → MaxPooling  
- Conv2D (256) → Conv2D (256) → MaxPooling  
- Flatten → Dense(1500) → Dense(43 with softmax)  

Loss Function: `categorical_crossentropy`  
Optimizer: Adam with learning rate `0.0001`  

## 🚀 How to Use

### 1. 📌 Train the Model (Optional)

Open `train_traffic_sign_recognition.ipynb` in Google Colab and run it step-by-step. It will:

- Mount Google Drive  
- Extract datasets from ZIP  
- Load data using `image_dataset_from_directory`  
- Train for 10 epochs  
- Save the model as `trained_model.h5`  
- Save training history as `training_hist.json`  
- Plot training/validation accuracy  

If you already have `trained_model.h5`, skip to step 2.

### 2. 🔍 Run the Streamlit App

- Install required packages:
- pip install streamlit tensorflow numpy
- Run the app:
- streamlit run main.py

## 🛠 Possible Improvements
- Add webcam-based real-time detection
- Show confidence scores
- Add Grad-CAM to visualize model attention
- Deploy to Hugging Face Spaces or Docker

## 🤝 Acknowledgements
- GTSRB Dataset
- TensorFlow/Keras
- Streamlit
- Google Colab

## 👨‍💻 Author
Muhammad Marzan Hussain
Final Year BSc in Computer Science
Developed as part of academic deep learning project