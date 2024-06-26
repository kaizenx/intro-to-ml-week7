from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load your data
file_path = '/path/to/loan_default.csv'
data = pd.read_csv(file_path)

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

# Step 5: Ensure the target variable 'Status' is categorical
data['Status'] = data['Status'].astype(int)

# Step 6: Select important features using a RandomForestClassifier
X = data.drop(columns=['Status'])
y = data['Status']

model = RandomForestClassifier()
model.fit(X, y)
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

# Creating the final dataframe
final_data = data.copy()

# Display the cleaned dataframe
final_data.head()




print(final_data.head())

final_data.to_csv('cleaned_loan_data_2.csv', index=False)
