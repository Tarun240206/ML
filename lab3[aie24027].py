import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



#1
a = np.array([2, 3, 4])
b = np.array([7, 10, 2])

def dot(x, y):
    s = 0
    if len(x) != len(y):
        print("vectors have different dimensions")
        return
    else:
        for i in range(len(x)):
            s += x[i] * y[i]

    d = np.dot(x, y)

    if d == s:
        print("values are same:", s)
    else:
        print("Invalid")

dot(a, b)

def norm(u):
    total = 0
    for i in range(len(u)):
        total += u[i] * u[i]

    sq = m.sqrt(total)
    n = np.linalg.norm(u)

    if n == sq:
        print("Norm values are same:", sq)
    else:
        print("Invalid")

norm(a)
norm(b)

#2
df = pd.read_excel(
    r"C:\Users\ADMIN\Downloads\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

print(df.head())

c1_feat_vecs = df[df["Customer"] == "C_1"].iloc[:, 1:].values
c2_feat_vecs = df[df["Customer"] == "C_2"].iloc[:, 1:].values

print("C1 Feature Matrix:")
print(c1_feat_vecs)
print("\nC2 Feature Matrix:")
print(c2_feat_vecs)

def cal_mean(a):
    total = 0
    for i in a:
        total += i
    return total / len(a)

def cal_var(a):
    mu = cal_mean(a)
    total = 0
    for i in a:
        total += (i - mu) ** 2
    return total / len(a)

def cal_std(a):   
    return m.sqrt(cal_var(a))


def dataset(matrix):
    means = []
    variances = []
    stds = []

    for i in range(matrix.shape[1]):
        feature = matrix[:, i]
        means.append(cal_mean(feature))
        variances.append(cal_var(feature))
        stds.append(cal_std(feature))

    return np.array(means), np.array(variances), np.array(stds)


c1_mean, c1_var, c1_std = dataset(c1_feat_vecs)
c2_mean, c2_var, c2_std = dataset(c2_feat_vecs)

print("C1  Mean:", c1_mean)
print("C1  Variance:", c1_var)
print("C1  Std Dev:", c1_std)

print("\nC2  Mean:", c2_mean)
print("C2  Variance:", c2_var)
print("C2  Std Dev:", c2_std)

centroid1 = c1_feat_vecs.mean(axis=0)
centroid2 = c2_feat_vecs.mean(axis=0)

print("C1 Centroid Vector:", centroid1)
print("C2 Centroid Vector:", centroid2)

print("C1 Std Dev :", c1_feat_vecs.std(axis=0))
print("C2 Std Dev :", c2_feat_vecs.std(axis=0))

Edist = np.linalg.norm(centroid1 - centroid2)
print("Euclidean Distance :", Edist)

#3
candies_feature = df.iloc[:,1].values
print("Candies feature: ",candies_feature)
histogram_candies_data = np.histogram(candies_feature)
mean_candies_feature = np.mean(candies_feature)
variance_candies_feature = np.var(candies_feature)
plt.hist(candies_feature)
plt.xlabel("Candies Purchased")
plt.ylabel("Frequency")
plt.show()
print("Histogram counts:", histogram_candies_data)
print("Mean of candies feature:", mean_candies_feature)
print("Variance of candies feature:", variance_candies_feature)

#4
def minkwosi_distance(feature_1,feature_2,p):
    diff=0
    result=0
    for i in range(len(feature_1)):
        if feature_1[i] > feature_2[i]:
           diff = feature_1[i] - feature_2[i]
        else:
           diff = feature_2[i] - feature_1[i]
        result += diff ** p
    return result ** (1/p)

p_vector = [1,2,3,4,5,6,7,8,9,10]
print(p_vector)

mangoes_feature_vector = df.iloc[:,2].values
milk_packets_feature_vector = df.iloc[:,3].values

minkwosi_dist_vector = []
for i in range(1,11):
   minkwosi_dist = minkwosi_distance(mangoes_feature_vector,milk_packets_feature_vector,i)
   minkwosi_dist_vector.append(minkwosi_dist)

print(minkwosi_dist_vector)
plt.plot(p_vector, minkwosi_dist_vector)
plt.xlabel("p(1-10)")
plt.ylabel("distance")
plt.show()


#5
print("Minkwosi distance through my function: ",minkwosi_distance(mangoes_feature_vector,milk_packets_feature_vector,2))
print("Minkwosi distance through python function: ",minkowski(mangoes_feature_vector, milk_packets_feature_vector, 2))

#6
X = df.iloc[:, 1:4].values
y = np.where(df.iloc[:, 4].values < 250, 0, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Training features: ", X_train)
print("Testing features: ", X_test)
print("Training labels: ", y_train)
print("Testing labels: ", y_test)

#7
neigh3 = KNeighborsClassifier(n_neighbors=3)
neigh3.fit(X_train, y_train)

#8
accuracy3 = neigh3.score(X_test, y_test)
print("Accuracy(k=3):", accuracy3)


#9
X_pred=neigh3.predict(X_test)
print("Predicted labels:", X_pred)

#10

def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return sqrt(distance)


def knn_custom(X_train, y_train, X_test, k):
    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], X_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for (_, label) in distances[:k]]

    predicted_class = Counter(k_nearest_labels).most_common(1)[0][0]
    return predicted_class

custom_predictions = []
for test_vector in X_test:
    pred = knn_custom(X_train, y_train, test_vector, k=3)
    custom_predictions.append(pred)

custom_predictions = np.array(custom_predictions)


correct = np.sum(custom_predictions == y_test)
custom_accuracy = correct / len(y_test)

print("Custom kNN Predictions:", custom_predictions)
print("Custom kNN Accuracy:", custom_accuracy)

print("Comparison of kNN Classifiers:")
print("Package kNN Accuracy:", accuracy3)
print("Custom kNN Accuracy:", custom_accuracy)

#11
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
accuracy = neigh.score(X_test, y_test)
print("Accuracy(k=1):", accuracy)
k=range(1,8)
accuracies=[]
for i in k:
    n=KNeighborsClassifier(n_neighbors=i)
    n.fit(X_train, y_train)
    acc=n.score(X_test, y_test)
    accuracies.append(acc)

plt.plot(k,accuracies,marker='o')
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("kNN Accuracy vs k")
plt.grid(True)
plt.show()

#12
X = df.iloc[:, 1:4].values
y = np.where(df.iloc[:, 4].values < 250, 0, 1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix (Training Data):")
print(cm_train)

print("\nConfusion Matrix (Testing Data):")
print(cm_test)


print("\nTraining Performance Metrics:")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("F1-Score:", f1_score(y_train, y_train_pred))


print("\nTesting Performance Metrics:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1-Score:", f1_score(y_test, y_test_pred))

#13

def confusion_matrix_custom(y_true, y_pred):
    TP = FP = TN = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    return TP, FP, TN, FN


def accuracy_custom(TP, FP, TN, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def precision_custom(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall_custom(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def fbeta_score_custom(precision, recall, beta):
    if precision + recall == 0:
        return 0
    return (1 + beta*2) * (precision * recall) / ((beta*2 * precision) + recall)


TP, FP, TN, FN = confusion_matrix_custom(y_test, y_test_pred)

print("Custom Confusion Matrix Values:")
print("TP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)

accuracy = accuracy_custom(TP, FP, TN, FN)
precision = precision_custom(TP, FP)
recall = recall_custom(TP, FN)
f1_score_custom = fbeta_score_custom(precision, recall, beta=1)

print("\nCustom Performance Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score_custom)






