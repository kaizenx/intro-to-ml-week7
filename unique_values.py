import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv("loan_default.csv")
# Select only numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Create a dictionary of unique values for numeric columns
unique_values_dict = {col: numeric_data[col].unique() for col in numeric_data.columns}

# Print unique values for each numeric column
for col, unique_vals in unique_values_dict.items():
    has_negative = (numeric_data[col] < 0).any()
    # print(f"Unique values in {col}: {unique_vals}")
    if has_negative:
        print(f"Column {col} contains negative values.")
    # else:
        # print(f"Column {col} does not contain negative values.")

# # Specify the column name
# column_name = 'Credit_Score'

# # Check if the column exists in the DataFrame
# if column_name in data.columns:
#     # Get unique values
#     unique_values = data[column_name].unique()
#     sorted_unique_values = sorted(unique_values)
#     print(sorted_unique_values)

#     summary_stats = data[column_name].describe()
#     print(f"Summary statistics for the column '{column_name}':")
#     print(summary_stats)

#      # QQ plot
#     plt.figure(figsize=(10, 6))
#     stats.probplot(data[column_name].dropna(), dist="norm", plot=plt)
#     plt.title(f'QQ Plot for {column_name}')
#     plt.show()

   
# else:
#     print(f"The column '{column_name}' does not exist in the dataset.")
