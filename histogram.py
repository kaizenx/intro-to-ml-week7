import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data
data = pd.read_csv("filtered_data.csv")

# Step 1: Preview the dataset
print(data.head())
print(data.info())

# Step 2: Summarize the dataset
print(data.describe())

# Filter the dataframe to include only numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64']).drop(columns=['Status'])
# use this one if you want to see status in the correlation matrix
# numeric_data = data.select_dtypes(include=['int64', 'float64'])

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

# sns.histplot(numeric_data["income"], kde=True, color='lightgreen', edgecolor='red')
# plt.xlabel('Values')
# plt.ylabel('Density')
# plt.title(f'Histogram with Density Plot for income')
# plt.xlim(numeric_data["income"].min(),numeric_data["income"].max())
# plt.tight_layout()
# plt.show()

# sns.histplot(numeric_data["LTV"], kde=True, color='lightgreen', edgecolor='red')
# plt.xlabel('Values')
# plt.ylabel('Density')
# plt.title(f'Histogram with Density Plot for LTV')
# plt.xlim(numeric_data["LTV"].min(),numeric_data["LTV"].max())
# plt.tight_layout()
# plt.show()



# Correlation analysis to detect high correlation 
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix["Status"].sort_values(ascending=False)
top_corr_features = correlation_with_target.index[1:9]  # Select top 8 correlated features
top_corr_features = top_corr_features.insert(0, "Status")
filtered_corr_matrix = correlation_matrix.loc[top_corr_features, top_corr_features]

print(correlation_with_target.to_string())

plt.figure(figsize=(14, 12))
sns.heatmap(filtered_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=.5)
plt.show()




