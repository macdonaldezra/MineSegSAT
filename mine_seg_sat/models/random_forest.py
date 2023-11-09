import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split

# Load data
# Replace with the correct file path and method to load data
data = np.load('data_file.npy')
X = data[:, :-1]  # All columns except the last one
y = data[:, -1]   # Only the last column

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
map_score = average_precision_score(y_test, y_pred, average='macro')

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Mean Average Precision: {map_score:.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Fine-tuning the model using Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)


# Make predictions using the fine-tuned model
y_pred_best = best_clf.predict(X_test)

# Evaluate the fine-tuned model
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, average='macro')
recall_best = recall_score(y_test, y_pred_best, average='macro')
map_score_best = average_precision_score(y_test, y_pred_best, average='macro')

# Print evaluation metrics for the fine-tuned model
print(f"Accuracy (Fine-tuned): {accuracy_best:.4f}")
print(f"Precision (Fine-tuned): {precision_best:.4f}")
print(f"Recall (Fine-tuned): {recall_best:.4f}")
print(f"Mean Average Precision (Fine-tuned): {map_score_best:.4f}")
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))

# Feature importance
best_clf = grid_search.best_estimator_
importances = best_clf.feature_importances_
for feature, importance in zip(['NDVI', 'NBR'], importances):
    print(f"Feature: {feature}, Importance: {importance}")
