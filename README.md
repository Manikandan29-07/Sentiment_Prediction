# Sentiment Analysis of Movie Reviews 🎬

## Project Overview

This project is an end-to-end machine learning pipeline for classifying the sentiment of movie reviews as either **positive** or **negative**. The model is trained on the IMDb movie review dataset and uses a **Logistic Regression** classifier with **TF-IDF** vectorization.

The project is structured into a modular and reproducible pipeline, covering all the essential steps of a standard machine learning workflow: data preprocessing, model training, evaluation, and live prediction.

## Key Features

* **Modular Code**: The project is organized into separate Python scripts for each stage of the pipeline (`data_preprocessing.py`, `model_training.py`, etc.), making the code clean, readable, and easy to maintain.
  
* **Data Preprocessing**: Includes robust text cleaning techniques like lowercasing, HTML tag removal, and lemmatization to prepare text data for the model.

* **TF-IDF Vectorization**: Converts raw text data into a numerical format that the machine learning model can understand.

* **Model Evaluation**: Provides a detailed analysis of the model's performance using metrics like Accuracy, Precision, Recall, and a Confusion Matrix.

* **Live Prediction**: A simple script to use the trained model for predicting the sentiment of new, unseen reviews.

## 📂 Project Structure

```text
sentiment-analysis-project/
├── IMDB Dataset.csv            # The raw dataset
├── data_preprocessing.py       # Functions for text cleaning and prep
├── model_training.py           # Logic for vectorization and training
├── model_evaluation.py         # Script to evaluate model performance
├── prediction.py               # Script for making live predictions
├── main.py                     # The main orchestrator script
└── trained_models/             # Directory to save the trained model
    ├── sentiment_model.pkl
    ├── tfidf_vectorizer.pkl
    └── label_encoder.pkl
```

## 🚀 How to Run the Project

### 1. Prerequisites
Make sure you have Python installed.  
It's recommended to create a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2. Install Libraries
Install all the required libraries using `pip`
```bash
pip install pandas scikit-learn nltk joblib seaborn numpy
```

### 3. Download the Dataset
Download the IMDb Dataset of 50k Movie Reviews from Kaggle and place the `IMDB Dataset.csv` file in the project's root directory.

### 4. Run the pipeline
The `main.py` script will execute the entire pipeline, from data preparation to model training, evaluation, and saving the final model files.
```bash
python main.py
```
This will output the evaluation results and save the trained model artifacts to the `trained_models/` directory.

## Evaluation Results

The model achieved the following performance on the test set:

### Accuracy: 88.48%

### Classification Report

```bash
              precision    recall  f1-score   support

           0       0.89      0.87      0.88      4961
           1       0.88      0.90      0.89      5039

    accuracy                           0.88     10000
   macro avg       0.89      0.88      0.88     10000
weighted avg       0.89      0.88      0.88     10000
```

### Confusion Matrix
```bash
[[4324  637]
 [ 515 4524]]
```

## Make a Live Prediction
You can use the `prediction.py` script to test the model with a new review.
```bash
python prediction.py
```
The script will demonstrate predictions on example reviews. Feel free to modify the `if __name__ == "__main__": ` block to test your own reviews.
