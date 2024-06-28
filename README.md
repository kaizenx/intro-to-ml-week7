# Supervised Machine Learning Project (Week 7)

## Source data
1. Source of loan data [here](https://www.kaggle.com/datasets/yasserh/loan-default-dataset/data)

## File descriptions

1. preprocessing.py is used to clean and prepare the data, it makes filtered_data.csv
2. histogram.py is used to create histograms and correlation matrix
3. unique_values.py is used to display unique values of the features and generate QQ plots for features
4. multicollinearity_check.py is used to calculate VIF and detect multicollinearity among features
5. describe.py was used to describe the data, get number of rows, features etc..
6. get_numeric_features.py is used to determine the numeric features
7. perfect_qq.py is used a helper to generate a perfect looking QQ chart
8. correlation_matrix.py is used to generate a correlation matrix of the filtered data
9. model_dummy.py is the dummy classifier implementation used for benchmark purposes
10. (Takes a while to run) model_logistic_regression.py is the logistic regression classifier implementation
11. (Takes a while to run) model_randomforest.py is the random forest implementation

## Instructions

1. You can start by running pip install with requirement.txt
2. Then download the data and rename the file to loadn_data.csv if need be.
3. Run munging.py to create the filtered_data.csv before using any of the other files.
4. Then based on your desire, you can run the other script files. For example run model_randomforest.py if you want to see the results for randomforest.
