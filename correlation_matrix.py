import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# set a column's datatype as categorical or numeric

# Load the data
data = pd.read_csv("filtered_data.csv")

# Check for infinite values and replace them with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill any NaNs that may have been introduced by the log transformation
for column in data.columns:
    if data[column].dtype.name != 'category':
        data[column] = data[column].fillna(data[column].median())

# Encode the target variable 'Status' as numeric.
# data['Status'] = data['Status'].astype(int)
data['Status'] = data['Status'].astype('category').cat.codes

# Correlation analysis to detect high correlation 
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix["Status"].sort_values(ascending=False)
top_corr_features = correlation_with_target.index[1:11]  # Select top 10 correlated features
filtered_corr_matrix = correlation_matrix.loc[top_corr_features, top_corr_features]

plt.figure(figsize=(14, 12))
sns.heatmap(filtered_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=.5)
plt.show()




