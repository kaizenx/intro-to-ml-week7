import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier


# Load the data
data = pd.read_csv("filtered_data.csv")

data['Status'] = data['Status'].astype('category').cat.codes

X = data.drop(columns=['Status'])
y = data['Status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 6: select important features 

# Method 1: Correlation Analysis
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix["Status"].sort_values(ascending=False)
top_corr_features = correlation_with_target.index[1:11]  # Select top 10 correlated features

print(top_corr_features)


# Define the dummy classifier model
dummy = DummyClassifier(strategy="uniform", random_state=42)

# Perform cross-validation with dummy classifier
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dummy_cv_results = cross_val_score(dummy, X[top_corr_features], y, cv=cv, scoring='accuracy')

print(f"Dummy Classifier Cross-Validation Accuracy Scores: {dummy_cv_results}")
print(f"Mean Dummy Classifier Cross-Validation Accuracy: {dummy_cv_results.mean()}")
print(f"Standard Deviation of Dummy Classifier Cross-Validation Accuracy: {dummy_cv_results.std()}")

# Train the dummy classifier
dummy.fit(X_train[top_corr_features], y_train)

# Make predictions with dummy classifier
y_dummy_pred = dummy.predict(X_test[top_corr_features])

# Evaluate the dummy classifier model
dummy_accuracy = accuracy_score(y_test, y_dummy_pred)
dummy_precision = precision_score(y_test, y_dummy_pred)
dummy_recall = recall_score(y_test, y_dummy_pred)
dummy_f1 = f1_score(y_test, y_dummy_pred)
dummy_roc_auc = roc_auc_score(y_test, y_dummy_pred)

print(f"Dummy Classifier Accuracy: {dummy_accuracy}")
print(f"Dummy Classifier Precision: {dummy_precision}")
print(f"Dummy Classifier Recall: {dummy_recall}")
print(f"Dummy Classifier F1 Score: {dummy_f1}")
print(f"Dummy Classifier ROC AUC Score: {dummy_roc_auc}")

# Display the confusion matrix for dummy classifier
dummy_conf_matrix = confusion_matrix(y_test, y_dummy_pred)
print("Confusion Matrix (Dummy Classifier):")
print(dummy_conf_matrix)

# Display the classification report for dummy classifier
dummy_class_report = classification_report(y_test, y_dummy_pred)
print("Classification Report (Dummy Classifier):")
print(dummy_class_report)

# Plot the confusion matrix for dummy classifier
sns.heatmap(dummy_conf_matrix, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Dummy Classifier)')
plt.show()