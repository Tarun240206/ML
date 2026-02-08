import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#1
path = r"c:\Users\ADMIN\Downloads\writer_identification_through_text_blogs_curated.xlsx"
df = pd.read_excel(path)
y = df.iloc[:, 200]
X = df.iloc[:, 0:200]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model = LinearSVC()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
print("Confusion matrix for training data:")
print(confusion_matrix(y_train, y_train_pred))
print("Classification reprto for training data(precision,recall and F1-score):")
print(classification_report(y_train, y_train_pred))
train_score = accuracy_score(y_train, y_train_pred)
print("Accuracy for training data:", train_score)

y_test_pred = model.predict(X_test)
print("Confusion Matrix for testing data:")
print(confusion_matrix(y_test, y_test_pred))
print("Classification Report for testing data:")
print(classification_report(y_test, y_test_pred))
test_score = accuracy_score(y_test, y_test_pred)
print("Accuracy for testing data:", test_score)

if train_score < 0.7 and test_score < 0.7:
    print("Model is underfit")
elif train_score > 0.9 and test_score < 0.7:
    print("Model is overfit")
else:
    print("Model is regular fit")

#2
path_2 = r"c:\Users\ADMIN\Downloads\Lab Session Data.xlsx"
df_2 = pd.read_excel(path_2)

X1 = df_2.iloc[:, 1:4]
y1 = df_2["Payment (Rs)"]
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X1, y1,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train_2, y_train_2)
y_pred_2 = model.predict(X_test_2)
mse = mean_squared_error(y_test_2, y_pred_2)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_2, y_pred_2)
r2 = r2_score(y_test_2, y_pred_2)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2:", r2)

#3
np.random.seed(42)
X = np.random.randint(1, 11, 20)
Y = np.random.randint(1, 11, 20)
labels = np.where(X + Y > 10, 1, 0)
colors = ["blue" if label == 0 else "red" for label in labels]
plt.scatter(X, Y, c=colors)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#4
np.random.seed(42)
X = np.random.randint(1, 11, 20)
Y = np.random.randint(1, 11, 20)
labels = np.where(X + Y > 10, 1, 0)
train_data = np.column_stack((X, Y))
x = np.arange(0, 10.1, 0.1)
y = np.arange(0, 10.1, 0.1)

xx, yy = np.meshgrid(x, y)
test_data = np.c_[xx.ravel(), yy.ravel()]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, labels)
pred = knn.predict(test_data)

colors_test = ["blue" if p == 0 else "red" for p in pred]
colors_train = ["blue" if l == 0 else "red" for l in labels]
plt.scatter(test_data[:,0], test_data[:,1], c=colors_test, s=10, alpha=0.4)
plt.scatter(X, Y, c=colors_train, s=80, edgecolors="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


#5
for i in range(4):
    test_data = np.c_[xx.ravel(), yy.ravel()]
    knn = KNeighborsClassifier(n_neighbors=(3+i))
    knn.fit(train_data, labels)
    pred = knn.predict(test_data)

    colors_test = ["blue" if p == 0 else "red" for p in pred]
    colors_train = ["blue" if l == 0 else "red" for l in labels]
    plt.scatter(test_data[:,0], test_data[:,1], c=colors_test, s=10, alpha=0.4)
    plt.scatter(X, Y, c=colors_train, s=80, edgecolors="black")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

#7
np.random.seed(42)
X = np.random.randint(1, 11, 20)
Y = np.random.randint(1, 11, 20)
labels = np.where(X + Y > 10, 1, 0)
train_data = np.column_stack((X, Y))
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': list(range(1, 16))
}
grid = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid.fit(train_data, labels)
best_k = grid.best_params_['n_neighbors']
print("Best k:", best_k)
print("Best Cross-Validation Accuracy:", grid.best_score_)
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(train_data, labels)


x = np.arange(0, 10.1, 0.1)
y = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x, y)
test_data = np.c_[xx.ravel(), yy.ravel()]
pred = best_knn.predict(test_data)

colors_test = ["blue" if p == 0 else "red" for p in pred]
colors_train = ["blue" if l == 0 else "red" for l in labels]
plt.figure(figsize=(8, 6))
plt.scatter(
    test_data[:, 0],
    test_data[:, 1],
    c=colors_test,
    s=10,
    alpha=0.4
)
plt.scatter(
    X,
    Y,
    c=colors_train,
    s=80,
    edgecolors="black"
)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
