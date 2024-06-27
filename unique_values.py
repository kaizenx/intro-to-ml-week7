import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv("filtered_data.csv")
# Specify the column name
column_name = 'Credit_Score'

# Check if the column exists in the DataFrame
if column_name in data.columns:
    # Get unique values
    unique_values = data[column_name].unique()
    sorted_unique_values = sorted(unique_values)
    print(sorted_unique_values)

    summary_stats = data[column_name].describe()
    print(f"Summary statistics for the column '{column_name}':")
    print(summary_stats)

     # QQ plot
    plt.figure(figsize=(10, 6))
    stats.probplot(data[column_name].dropna(), dist="norm", plot=plt)
    plt.title(f'QQ Plot for {column_name}')
    plt.show()

   
else:
    print(f"The column '{column_name}' does not exist in the dataset.")
