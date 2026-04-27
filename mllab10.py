import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import lime
import lime.lime_tabular
import shap


def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = pd.get_dummies(X)

    return X.values, y.values


# A1
def correlation_analysis(X):
    corr_matrix = pd.DataFrame(X).corr()
    return corr_matrix


# A2 (99% variance PCA)
def pca_99(X_train, X_test):
    pca = PCA(n_components=0.99)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


# A3 (95% variance PCA)
def pca_95(X_train, X_test):
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


# Model training
def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc


# A4 Sequential Feature Selection
def sequential_fs(X_train, X_test, y_train):
    model = LogisticRegression(max_iter=1000)

    cv=StratifiedKFold(n_splits=2)

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=10,
        direction="forward",
        cv=cv
    )

    sfs.fit(X_train, y_train)

    X_train_sfs = sfs.transform(X_train)
    X_test_sfs = sfs.transform(X_test)

    return X_train_sfs, X_test_sfs


# A5 LIME
def lime_explain(model, X_train, X_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        mode='classification'
    )

    explanation = explainer.explain_instance(
        X_test[0],
        model.predict_proba
    )

    explanation.save_to_file(r"C:/Users/ADMIN/Downloads/lime_explanation.html")


# A5 SHAP
def shap_explain(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train[:50])
    shap.summary_plot(shap_values, X_train[:50])


if __name__ == "__main__":
    file_path = r"C:\Users\ADMIN\Downloads\writer_identification_through_text_blogs_curated.xlsx"
    X, y = load_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # A1
    corr_matrix = correlation_analysis(X)
    plt.figure()
    sns.heatmap(corr_matrix)
    plt.show()

    # Original model
    base_model, base_acc = train_model(X_train, X_test, y_train, y_test)

    # A2
    X_train_99, X_test_99, _ = pca_99(X_train, X_test)
    model_99, acc_99 = train_model(X_train_99, X_test_99, y_train, y_test)

    # A3
    X_train_95, X_test_95, _ = pca_95(X_train, X_test)
    model_95, acc_95 = train_model(X_train_95, X_test_95, y_train, y_test)

    # A4
    X_train_sfs, X_test_sfs = sequential_fs(X_train, X_test, y_train)
    model_sfs, acc_sfs = train_model(X_train_sfs, X_test_sfs, y_train, y_test)

    # A5
    lime_explain(base_model, X_train, X_test)
    shap_explain(base_model, X_train)

    print("Original Accuracy:", base_acc)
    print("PCA 99% Accuracy:", acc_99)
    print("PCA 95% Accuracy:", acc_95)
    print("SFS Accuracy:", acc_sfs)