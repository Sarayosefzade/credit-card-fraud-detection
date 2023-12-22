import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

# load the dataset using pandas
data = pd.read_csv('creditcard.csv')

# Sample the dataset
data = data.sample(frac=0.1, random_state=1)

# Print the shape of the data
print(data.shape)

# Print basic statistics
print(data.describe())

data.hist(figsize=(20, 20))
plt.show()

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))
print('Outlier fraction:', outlier_fraction)

print('Fraud Cases:', len(Fraud))
print('Valid Transactions:', len(Valid))

corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

columns = [c for c in data.columns if c not in ['Class']]
X = data[columns]
Y = data['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

smote = SMOTE(random_state=2)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Logistic Regression
logistic_model = LogisticRegression(class_weight='balanced', solver='liblinear')
logistic_model.fit(X_train_resampled, Y_train_resampled)

Y_probs = logistic_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(Y_test, Y_probs)
threshold = thresholds[np.argmax(2*recall[:-1]*precision[:-1]/(recall[:-1]+precision[:-1]))]

Y_pred = (Y_probs >= threshold).astype(int)

# Printing the classification report
print(classification_report(Y_test, Y_pred))

# Printing the F1 score
print('F1 Score:', f1_score(Y_test, Y_pred))

print('Accuracy Score:', accuracy_score(Y_test, Y_pred))
