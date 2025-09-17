from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model(X_train, y_train):
    """
    Trains a TfidfVectorizer and a Logistic Regression model within a pipeline.

    Args:
        X_train (pd.Series): The training text data.
        y_train (pd.Series): The training labels.

    Returns:
        tuple: A tuple containing the fitted TfidfVectorizer and the trained LogisticRegression model.
    """
    
    # Initialize the TfidVectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit the vectorizer on the training data - training data is get transformed into vector form so that it can be fed into the model
    print("Fitting TfidVectorizer on training data")
    training_vectorized = vectorizer.fit_transform(X_train)
    print("TfidVectorizer fitted")
    
    # Setting up the model
    model = LogisticRegression(max_iter=1000)
    model.fit(training_vectorized, y_train)
    print("Training finished")
    
    return vectorizer, model



# if __name__ == '__main__':
#     # This block is for testing the function independently
#     from sklearn.model_selection import train_test_split

#     # Create some dummy data
#     sample_reviews = [
#         "This is a great movie, I loved it!",
#         "The film was terrible and boring.",
#         "I didn't like the ending, but the acting was okay.",
#         "A masterpiece of cinema, absolutely fantastic."
#     ]
#     sample_sentiments = ['positive', 'negative', 'neutral', 'positive']
    
#     # Split data (just for a simple test)
#     X_train, X_test, y_train, y_test = train_test_split(
#         sample_reviews, sample_sentiments, test_size=0.5, random_state=42
#     )

#     # Train the model and get back the vectorizer and model
#     tfidf_vectorizer, trained_model = train_model(X_train, y_train)

#     # To show it works, let's predict on the test data
#     X_test_vectorized = tfidf_vectorizer.transform(X_test)
#     predictions = trained_model.predict(X_test_vectorized)
    
#     print("\n--- Independent Test Results ---")
#     print(f"Test reviews: {X_test}")
#     print(f"Predicted sentiments: {predictions}")
#     print(f"Actual sentiments: {y_test}")