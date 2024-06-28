import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier

# Load the data
data = pd.read_csv("filtered_data.csv")

data['Status'] = data['Status'].astype('category').cat.codes

# selected features
selected_features = [ 'credit_type_EQUI', 'co-applicant_credit_type_EXP',
                      'submission_of_application_to_inst', 'term_300.0', 'dtir1' ]

X = data[selected_features]
y = data['Status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Define the dummy classifier model
dummy = DummyClassifier(strategy="uniform", random_state=100)

# Perform cross-validation with dummy classifier
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
dummy_cv_results = cross_val_score(dummy, X, y, cv=cv, scoring='accuracy')

print(f"Dummy Classifier Cross-Validation Accuracy Scores: {dummy_cv_results}")
print(f"Mean Dummy Classifier Cross-Validation Accuracy: {dummy_cv_results.mean()}")
print(f"Standard Deviation of Dummy Classifier Cross-Validation Accuracy: {dummy_cv_results.std()}")

# Train the dummy classifier
dummy.fit(X_train, y_train)

# Make predictions with dummy classifier on test data
y_dummy_test_pred = dummy.predict(X_test)

# Evaluate the dummy classifier model on test data
dummy_test_accuracy = accuracy_score(y_test, y_dummy_test_pred)
dummy_test_precision = precision_score(y_test, y_dummy_test_pred)
dummy_test_recall = recall_score(y_test, y_dummy_test_pred)
dummy_test_f1 = f1_score(y_test, y_dummy_test_pred)
dummy_test_roc_auc = roc_auc_score(y_test, y_dummy_test_pred)

print(f"Dummy Classifier Test Accuracy: {dummy_test_accuracy}")
print(f"Dummy Classifier Test Precision: {dummy_test_precision}")
print(f"Dummy Classifier Test Recall: {dummy_test_recall}")
print(f"Dummy Classifier Test F1 Score: {dummy_test_f1}")
print(f"Dummy Classifier Test ROC AUC Score: {dummy_test_roc_auc}")

# Make predictions with dummy classifier on training data
y_dummy_train_pred = dummy.predict(X_train)

# Evaluate the dummy classifier model on training data
dummy_train_accuracy = accuracy_score(y_train, y_dummy_train_pred)
dummy_train_precision = precision_score(y_train, y_dummy_train_pred)
dummy_train_recall = recall_score(y_train, y_dummy_train_pred)
dummy_train_f1 = f1_score(y_train, y_dummy_train_pred)
dummy_train_roc_auc = roc_auc_score(y_train, y_dummy_train_pred)

print(f"Dummy Classifier Training Accuracy: {dummy_train_accuracy}")
print(f"Dummy Classifier Training Precision: {dummy_train_precision}")
print(f"Dummy Classifier Training Recall: {dummy_train_recall}")
print(f"Dummy Classifier Training F1 Score: {dummy_train_f1}")
print(f"Dummy Classifier Training ROC AUC Score: {dummy_train_roc_auc}")

# Display the confusion matrix for dummy classifier on test data
dummy_test_conf_matrix = confusion_matrix(y_test, y_dummy_test_pred)
print("Confusion Matrix (Dummy Classifier - Test Data):")
print(dummy_test_conf_matrix)

# Display the classification report for dummy classifier on test data
dummy_test_class_report = classification_report(y_test, y_dummy_test_pred)
print("Classification Report (Dummy Classifier - Test Data):")
print(dummy_test_class_report)

# Plot the confusion matrix for dummy classifier on test data
sns.heatmap(dummy_test_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Dummy Classifier - Test Data)')
plt.show()
