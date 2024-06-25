import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

data = pd.read_csv("loan_default.csv")



# Step 1: Handle missing values
# For simplicity, let's fill missing numerical values with the median and categorical with the mode
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].median(), inplace=True)

# Step 2: Encode categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Step 3: Create interaction features
data['Credit_Score_LTV'] = data['Credit_Score'] * data['LTV']

# Step 4: Normalize/scale numerical features
scaler = StandardScaler()
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 5: Select important features using a RandomForestClassifier
X = data.drop(columns=['Status'])
y = data['Status']

model = RandomForestClassifier()
model.fit(X, y)
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

# Creating the final dataframe
final_data = data.copy()

import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned Loan Data", dataframe=final_data)

# The cleaned dataframe is now ready for analysis
final_data.head()
