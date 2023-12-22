import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('creditcard.csv')

# Print the shape of the data
data = data.sample(frac=0.1, random_state=1)
print("Sampled data shape:", data.shape)

fraud_cases = data[data['Class'] == 1]
valid_cases = data[data['Class'] == 0]

print('Fraud Cases: {}'.format(len(fraud_cases)))
print('Valid Transactions: {}'.format(len(valid_cases)))

columns = [c for c in data.columns if c not in ["Class"]]
X = data[columns]
Y = data['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
svc_classifier = SVC(class_weight='balanced', random_state=1)

grid_search = GridSearchCV(svc_classifier, parameters, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, Y_train)

best_svc = grid_search.best_estimator_

y_pred = best_svc.predict(X_test)

# Output the classification report and F1 score
print("Best parameters found:", grid_search.best_params_)
print(classification_report(Y_test, y_pred))
print('F1 Score:', f1_score(Y_test, y_pred))
