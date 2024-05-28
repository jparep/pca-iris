import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.target_names

def create_pipeline():
    scaler = StandardScaler()
    pca = PCA()
    logistic = LogisticRegression(max_iter=10000)
    return Pipeline(steps=[('scaler', scaler), ('pca', pca), ('logistic', logistic)])

def perform_grid_search(pipeline, X, y):
    param_grid = {
        'pca__n_components': [2, 3, 4],
        'logistic__C': np.linspace(10, 25, 16)  # Testing more values around 20
    }
    search = GridSearchCV(pipeline, param_grid, cv=5)
    search.fit(X, y)
    return search

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

def plot_pca_results(pca, X, y, target_names):
    X_pca = pca.transform(X)
    plt.figure(figsize=(8, 6))
    for i, target_name in enumerate(target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)
    plt.title("PCA of Iris Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

def main():
    X, y, target_names = load_and_preprocess_data()
    pipeline = create_pipeline()
    search = perform_grid_search(pipeline, X, y)
    
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    
    best_model = search.best_estimator_
    best_model.fit(X, y)
    
    evaluate_model(best_model, X, y)
    plot_pca_results(best_model.named_steps['pca'], X, y, target_names)

if __name__ == "__main__":
    main()
