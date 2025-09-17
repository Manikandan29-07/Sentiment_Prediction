# Convert the raw data into the model-ready format

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder


try :
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

try : 
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')
    
# Initialize the lemmatizer and set of stop words

lemmatizer = WordNetLemmatizer()        # Converts words to their base form... Eg : running, ran, runs -> run
stop_words = set(stopwords.words('english'))

def preprocess_text(text : str) -> str: 
    """
    Cleans and preprocesses a single text string.

    Args:
        text (str): The raw text string.

    Returns:
        str: The cleaned and preprocessed text.
    """
    
    text = text.lower()                 # Lowercase the text
    text = re.sub(r'<.*?>', '', text)   # Remove HTML tags using a regular expression
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))    # Removing punctuation and numbers
    
    words = text.split()    # Removes extra white spaces and split into words
    
    #Lemmatize words and remove stop words
    preprocess_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join the words back to the string 
    return ' '.join(preprocess_words)

def prepare_data(df:pd.DataFrame, text_column: str, label_column: str):
    """
    Applies preprocessing to a DataFrame and encodes labels.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing text data.
        label_column (str): The name of the column containing labels.

    Returns:
        tuple: A tuple containing the preprocessed DataFrame and the fitted LabelEncoder.
    """
    
    print("Starting Data preprocessing")
    
    # Applying the preprocessing function to the specified text column
    df['clean_text'] = df[text_column].apply(preprocess_text)
    
    # Encode the sentiment labels ( eg positive as 1, negative as 0)
    le = LabelEncoder()
    df['encoded_labels'] = le.fit_transform(df[label_column])
    
    print("Data Preprocessing complete")
    return df, le


# if __name__ == '__main__':
#     # This block is for testing the functions independently
#     # Create a sample DataFrame for demonstration
#     sample_data = {
#         'review': [
#             "This movie was really good! It was the best film I've seen. <br /> I would watch it again.",
#             "I hated the film. It was so bad and boring. The acting was terrible. It's a 1/10.",
#             "This is a neutral review. It's neither good nor bad. Just average."
#         ],
#         'sentiment': ['positive', 'negative', 'neutral']
#     }
#     sample_df = pd.DataFrame(sample_data)

#     print("Original DataFrame:")
#     print(sample_df)
#     print("-" * 50)

#     # Prepare the sample data and see the output
#     prepared_df, label_encoder = prepare_data(sample_df.copy(), 'review', 'sentiment')
    
#     print("\nPrepared DataFrame:")
#     print(prepared_df)
#     print("\nLabel Mappings:")
#     # Show the mapping of original labels to encoded numbers
#     for i, label in enumerate(label_encoder.classes_):
#         print(f"  {label}: {i}")