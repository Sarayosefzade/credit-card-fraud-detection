import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy
from sklearn.metrics import f1_score


# load the dataset using pandas
data = pd.read_csv('creditcard.csv')

# dataset exploring
print(data.columns)


data = data.sample(frac=0.1, random_state = 1)
print(data.shape)
print(data.describe())



Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

columns = data.columns.tolist()

columns = [c for c in columns if c not in ["Class"]]

target = "Class"

X = data[columns]
Y = data[target]

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=1, n_init=10)
kmeans.fit(X_scaled)

labels = kmeans.labels_

data['cluster'] = labels

counts = np.bincount(labels)
print(counts)

print(pd.crosstab(data['Class'], labels))

from sklearn.metrics import silhouette_samples, silhouette_score

silhouette_avg = silhouette_score(X_scaled, labels)
print("For n_clusters =", 2, "The average silhouette_score is :", silhouette_avg)

fraud_cluster = np.argmax([np.sum(labels[data['Class'] == 1] == 0), 
                           np.sum(labels[data['Class'] == 1] == 1)])
predicted_classes = np.where(labels == fraud_cluster, 1, 0)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(data['Class'], predicted_classes)

# Printing the f1 score and accuracy
print(f'Accuracy Score: {accuracy}')