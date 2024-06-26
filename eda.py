import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("loan_default.csv")
unnecessary_features = ['ID']
print(data.head(10))
# unique_values_dict = {col: data[col].unique() for col in data.columns}

# for col, unique_vals in unique_values_dict.items():
#     print(f"Unique values in {col}: {unique_vals}")

nan_percentage = data.isna().mean() * 100

# Print the percentage of NaN values per column to determine the missing values, or percentage of missing values per column
print(nan_percentage)

data.drop(columns=unnecessary_features, inplace=True)





