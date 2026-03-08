import sys
print("Running from:", sys.executable)
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

#1
path = r"c:\Users\ADMIN\Downloads\Blog_TFIDF_Vectors.xlsx"
df = pd.read_excel(path)

X_single = df.iloc[:, [0]]    
y = df.iloc[:, 200]           

X_train, X_test, y_train, y_test = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

reg = LinearRegression().fit(X_train, y_train)

y_train_pred = reg.predict(X_train)

print("Predicted values on training data:")
print(y_train_pred)

#2
X_single = df.iloc[:, 0:50]    
y = df.iloc[:, 200]           

X_train, X_test, y_train, y_test = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

reg = LinearRegression().fit(X_train, y_train)

y_train_pred = reg.predict(X_train)

print("Predicted values on training data:")
print(y_train_pred)

#3
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
r2_test = r2_score(y_test, y_test_pred)

print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAPE:", mape_train)
print("R2:", r2_train)

print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAPE:", mape_test)
print("R2:", r2_test)

#4

X = df.drop(columns=['Author_Label'])

X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)

kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
kmeans.fit(X_train)

print("Cluster Labels:")
print(kmeans.labels_)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

#5
X = df.drop(columns=['Author_Label'])

X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X_train)

labels = kmeans.labels_

sil_score = silhouette_score(X_train, labels)
ch_score = calinski_harabasz_score(X_train, labels)
db_index = davies_bouldin_score(X_train, labels)

print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_index)

#6
X = df.drop(columns=['Author_Label'])

k_values = range(2, 11)

sil_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    labels = kmeans.labels_
    
    sil_scores.append(silhouette_score(X, labels))
    ch_scores.append(calinski_harabasz_score(X, labels))
    db_scores.append(davies_bouldin_score(X, labels))

plt.figure()
plt.plot(k_values, sil_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.show()

plt.figure()
plt.plot(k_values, ch_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Calinski-Harabasz Score")
plt.title("CH Score vs k")
plt.show()

plt.figure()
plt.plot(k_values, db_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Davies-Bouldin Index")
plt.title("DB Index vs k")
plt.show()

#7
X = df.drop(columns=['Author_Label'])

X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)

distortions = []

for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_train)
    distortions.append(kmeans.inertia_)

plt.figure()
plt.plot(range(2, 20), distortions, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.show()