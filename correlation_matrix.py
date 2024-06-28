import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# set a column's datatype as categorical or numeric

# Load the data
data = pd.read_csv("filtered_data.csv")

# Correlation analysis to detect high correlation 
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix["Status"].sort_values(ascending=False)
top_corr_features = correlation_with_target.index[1:6]  
top_corr_features = top_corr_features.insert(0, "Status")
filtered_corr_matrix = correlation_matrix.loc[top_corr_features, top_corr_features]

print(top_corr_features)

plt.figure(figsize=(14, 12))
sns.heatmap(filtered_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=.5)
plt.show()




