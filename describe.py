import pandas as pd
import os

def describe_dataset(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Get the basic information
    num_rows, num_cols = df.shape
    file_size = os.path.getsize(file_path)
    
    # Get data types summary
    data_types = df.dtypes.value_counts()

    # Describe key features
    key_features_description = df.describe(include='all').T

    # Determine if the data is from multiple sources
    # This is a placeholder as determining multiple sources might need domain-specific checks
    multi_source = "Unknown"  # Update this logic based on your specific dataset

    # Prepare the description
    description = {
        "Number of samples/rows": num_rows,
        "Number of features/columns": num_cols,
        "File size in bytes": file_size,
        "Data types summary": data_types.to_dict(),
        "Key features description": key_features_description.to_dict(),
        "Multi-table or multiple data sources": multi_source
    }
    
    return description

# Example usage
file_path = 'loan_default.csv'
dataset_description = describe_dataset(file_path)
print(dataset_description)
