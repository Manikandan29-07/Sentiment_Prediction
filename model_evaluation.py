from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, vectorizer, X_test, y_test):
    """
    Evaluates the trained model on the test data.

    Args:
        model: The trained machine learning model.
        vectorizer: The fitted TfidfVectorizer.
        X_test (pd.Series): The testing text data.
        y_test (pd.Series): The testing labels.
    """
    print("....Model Evaluating....")
    
    #Transfroming the test data to vector
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Make Prediction 
    y_predict = model.predict(X_test_vectorized)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_predict)
    print(f"ACCURACY : {accuracy:.4f}")
    
    print()
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_predict))
    
    print("CONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    
   
    # Plot the Confusion Matrix for better visualization
    # class_labels = np.unique(y_test)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.show()

    print("\nEvaluation complete.")
    