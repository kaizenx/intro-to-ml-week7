import pandas as pd
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("filtered_data.csv")

# Identify numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Apply PowerTransformer to all numeric columns
pt = PowerTransformer(method='yeo-johnson')
data[numeric_data] = pt.fit_transform(data[numeric_data])



# Plot histograms for each numeric feature in one chart
num_features = numeric_data.shape[1]
num_cols = 3
num_rows = (num_features + num_cols - 1) // num_cols

plt.figure(figsize=(15, num_rows * 5))

for i, feature in enumerate(numeric_data.columns, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.histplot(numeric_data[feature], bins=30, kde=True, color='lightgreen', edgecolor='red')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title(f'Histogram with Density Plot for {feature}')

plt.tight_layout()
plt.show()

# Save the resulting dataframe to a new file
# data.to_csv("transformed_filtered_data.csv", index=False)
