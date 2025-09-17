import joblib 
from data_preprocessing import preprocess_text

def predict_sentiment(review: str):
    """
    Predicts the sentiment of a given text string.

    Args:
        review (str): The input text string (e.g., a movie review).

    Returns:
        str: The predicted sentiment label ('positive' or 'negative').
    """
    try:
        # Load the trained model and other necessary components
        model = joblib.load("trained_models/sentiment_model.pkl")
        vectorizer = joblib.load("trained_models/tfidf_vectorizer.pkl")
        label_encoder = joblib.load("trained_models/label_encoder.pkl")
    except FileNotFoundError:
        return "Error: Trained model files not found. Please run main.py first."
    
    # 1 - Clean the text using the same function from preprocessing
    cleaned_text = preprocess_text(review)
    
    # 2 - Vectorize the cleaned text
    # The vectorizer expects a list of documents, so we pass `[cleaned_text]`
    review_vectorized = vectorizer.transform([cleaned_text])
    
    # 3 - Predict the numerical label
    prediction = model.predict(review_vectorized)
    
    # 4 - Convert the numerical label back to a human-readable string
    predicted_label = label_encoder.inverse_transform(prediction)
    
    # 5 - Return the single predicted label from the array
    return predicted_label[0]

if __name__ == "__main__":
    
    print("SENTIMENT PREDICTION")
    print()
    rev = "This movie was not fantastic! I won't  recommend it."
    sentiment = predict_sentiment(rev)
    print(f"{rev} : {sentiment}")