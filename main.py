import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from data_preprocessing import prepare_data
from model_training import train_model
from model_evaluation import evaluate_model

def main():
    """
    Main function to run the sentiment analysis pipeline.
    """
    # 1-Load and Prepare data
    try :
        df = pd.read_csv('IMDB Dataset.csv')
    except FileNotFoundError:
        print("The IMDB Dataset.csv not found")
        return

    # use prepare_data() 
    preprocessed_df, label_encoder = prepare_data(df, text_column='review', label_column='sentiment')
    
    # Saving tha label encoder for future use
    joblib.dump(label_encoder, 'trained_models/label_encoder.pkl')
    print("Label encoder Saved")
    
    # 2-Split data into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_df['clean_text'],
        preprocessed_df['encoded_labels'],
        test_size=0.2,
        random_state=42
    )
    
    # 3- Train the model
    print("Training the model")
    vectorizer, model = train_model(X_train, y_train)
    
    # 4- Evaluate the model
    print("Evaluate the model")
    evaluate_model(model, vectorizer, X_test, y_test)
    
    # 5- Saving the model and vectorizer
    print("Saving The Model and Vectorizer")
    joblib.dump(model, 'trained_models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'trained_models/tfidf_vectorizer.pkl')
    print("Saved models")
    
    
    
if __name__ == "__main__":
    # Create the directory to save the models if they dont exists
    import os
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    
    main()