import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = pd.read_csv("loan_default.csv")
unnecessary_features = ['ID']
print(data.head())

# unique_values_dict = {col: data[col].unique() for col in data.columns}

# for col, unique_vals in unique_values_dict.items():
#     print(f"Unique values in {col}: {unique_vals}")

data.drop(columns=unnecessary_features, inplace=True)

# set a column's datatype as categorical or numeric
categorical_columns = ['year', 'loan_limit', 'Gender', 'approv_in_adv', 
                       'loan_type', 'loan_purpose','Credit_Worthiness', 'open_credit',
                       'business_or_commercial','term','Neg_ammortization','interest_only',
                       'lump_sum_payment','term','Neg_ammortization','construction_type',
                       'occupancy_type','Secured_by','total_units','credit_type',
                       'co-applicant_credit_type','age','submission_of_application','Region',
                       'Security_Type'
                       ]

for column in data.columns:
    if column in categorical_columns:
        data[column] = data[column].astype('category')
    else:
        data[column] = pd.to_numeric(data[column], errors='coerce')

# Step 1: Handle missing values
# Fill missing values
for column in data.columns:
    if data[column].dtype.name == 'category':
        data[column] = data[column].fillna(data[column].mode()[0])
    else:
        data[column] = data[column].fillna(data[column].median())
        

# Step 2: Encode categorical variables using One-Hot
# data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

data['Status'] = data['Status'].astype(int)

# Step 3: Create interaction features
# data['Credit_Score_LTV'] = data['Credit_Score'] * data['LTV']

# Step 4: Normalize/scale numerical features
# scaler = StandardScaler()
# numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
# data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

X = data.drop(columns=['Status'])
y = data['Status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 6: select important features 

# # Method 1: Correlation Analysis
# correlation_matrix = data.corr()
# correlation_with_target = correlation_matrix["Status"].sort_values(ascending=False)
# top_corr_features = correlation_with_target.index[1:11]  # Select top 10 correlated features



# Method 3: Gradient Boosting
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
gboost_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_})
gboost_features = gboost_importance.sort_values(by='Importance', ascending=False).head(10)['Feature']

# # Method 4: RandomForestClassifier
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
# rf_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
# rf_features = rf_importance.sort_values(by='Importance', ascending=False).head(10)['Feature']


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Print evaluation metrics
print("Training Set Performance:")
print(f'Accuracy: {train_accuracy}')
print(f'Precision: {train_precision}')
print(f'Recall: {train_recall}')
print(f'F1 Score: {train_f1}')

print("\nTest Set Performance:")
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {test_f1}')

# Cross-validation
cross_val_scores = cross_val_score(model, X, y, cv=5)
print("\nCross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", cross_val_scores.mean())

# print(results_df)

# # Creating the final dataframe
# final_data = data.copy()

# # Display the cleaned dataframe
# final_data.head()

# print(final_data.head())

# final_data.to_csv('cleaned_loan_data_2.csv', index=False)
