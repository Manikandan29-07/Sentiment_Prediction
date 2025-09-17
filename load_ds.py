import pandas as pd
df = pd.read_csv("IMDB Dataset.csv")

# print(df.head())   # Looking at the first few rows
# print(df.shape)      # Shape
# print(df.isnull().sum()) #Check for missing values
print(df['sentiment'].value_counts())