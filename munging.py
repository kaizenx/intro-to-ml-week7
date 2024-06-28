import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np

data = pd.read_csv("loan_default.csv")
unique_values_dict = {col: data[col].unique() for col in data.columns}

for col, unique_vals in unique_values_dict.items():
    print(f"Unique values in {col}: {unique_vals}")

nan_percentage = data.isna().mean() * 100


print(nan_percentage)
unnecessary_features = ['ID','year']
print(data.head())

data.drop_duplicates(inplace=True)
data.drop(columns=unnecessary_features, inplace=True)
data = data.dropna(how='all')

# set a column's datatype as categorical or numeric
categorical_columns = ['loan_limit', 'Gender', 'approv_in_adv', 
                       'loan_type', 'loan_purpose','Credit_Worthiness', 'open_credit',
                       'business_or_commercial','term','Neg_ammortization','interest_only',
                       'lump_sum_payment','Neg_ammortization','construction_type',
                       'occupancy_type','Secured_by','total_units','credit_type',
                       'co-applicant_credit_type','age','submission_of_application','Region',
                       'Security_Type'
                       ]
numerical_columns = ['loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'property_value', 'income', 'Credit_Score', 'LTV', 'dtir1']

# missing values with mode
for column in categorical_columns:
    if column in data.columns:
        data[column] = data[column].fillna(data[column].mode()[0])

# missing values with median
for column in numerical_columns:
    if column in data.columns:
        data[column] = data[column].fillna(data[column].median())

# Replace outliers with median
for column in numerical_columns:
    if column in data.columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median = data[column].median()
        data[column] = np.where((data[column] < lower_bound) | (data[column] > upper_bound), median, data[column])

    
# Use one-hot encoding to convert categorical variables into numerical format.
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Encode the target variable 'Status' as numeric.
data['Status'] = data['Status'].astype(int)


# Normalize/scale numerical features
scaler = RobustScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

data['Status'] = data['Status'].astype('category').cat.codes

data.to_csv("filtered_data.csv", index=False)
