import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset
data = pd.read_csv("filtered_data.csv")

# Encode the target variable if necessary
data['Status'] = data['Status'].astype('category').cat.codes

# Create a pairplot with hue based on the target variable
sns.pairplot(data, hue='Status')
plt.show()