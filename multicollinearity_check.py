import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the data
data = pd.read_csv("filtered_data.csv")

# Create a DataFrame with the relevant features
X = data[["credit_type_EQUI", "co-applicant_credit_type_EXP"]]
# X = data.select_dtypes(include=['int64', 'float64'])


# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(features_with_const.values, i) for i in range(features_with_const.shape[1])]
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 

print(vif_data)
