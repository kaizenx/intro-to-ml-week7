import pandas as pd

# Define the list of categorical columns
categorical_columns = ['loan_limit', 'Gender', 'approv_in_adv', 
                       'loan_type', 'loan_purpose','Credit_Worthiness', 'open_credit',
                       'business_or_commercial','term','Neg_ammortization','interest_only',
                       'lump_sum_payment','Neg_ammortization','construction_type',
                       'occupancy_type','Secured_by','total_units','credit_type',
                       'co-applicant_credit_type','age','submission_of_application','Region',
                       'Security_Type'
                       ]

# Load the CSV file
df = pd.read_csv('loan_default.csv')

# Get the remaining feature names by excluding the categorical columns
remaining_columns = [col for col in df.columns if col not in categorical_columns]

# Print the remaining feature names
print("Remaining feature names:")
print(remaining_columns)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print(numerical_columns)
