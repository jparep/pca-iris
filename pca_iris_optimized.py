import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.target_names

def create_pipeline(model):
    scaler = StandardScaler()
    pca = PCA()
    return Pipeline(steps=[('scaler', scaler), ('pca', pca), ('classifier', model)])

def perform_grid_search(pipeline, X, y, param_grid):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy')
    search.fit(X, y)
    return search

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
        print("\nROC AUC Score: {:.3f}".format(roc_auc_score(y, y_prob, multi_class='ovo')))

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

    # Define individual models
    logistic = LogisticRegression(max_iter=10000)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    # Create pipelines for each model
    logistic_pipeline = create_pipeline(logistic)
    random_forest_pipeline = create_pipeline(random_forest)

    # Define parameter grids for hyperparameter tuning
    param_grid_logistic = {
        'pca__n_components': [2, 3, 4],
        'classifier__C': np.linspace(10, 25, 16)
    }

    param_grid_rf = {
        'pca__n_components': [2, 3, 4],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30]
    }

    # Perform grid search for each model
    logistic_search = perform_grid_search(logistic_pipeline, X, y, param_grid_logistic)
    rf_search = perform_grid_search(random_forest_pipeline, X, y, param_grid_rf)

    # Print best parameters and scores
    print("Logistic Regression Best parameter (CV score=%.3f):" % logistic_search.best_score_)
    print(logistic_search.best_params_)
    print("Random Forest Best parameter (CV score=%.3f):" % rf_search.best_score_)
    print(rf_search.best_params_)

    # Combine the best models using Voting Classifier
    best_logistic = logistic_search.best_estimator_
    best_rf = rf_search.best_estimator_

    ensemble = VotingClassifier(estimators=[
        ('logistic', best_logistic),
        ('rf', best_rf)
    ], voting='soft')

    ensemble.fit(X, y)

    # Evaluate the ensemble model
    evaluate_model(ensemble, X, y)
    plot_pca_results(best_logistic.named_steps['pca'], X, y, target_names)

if __name__ == "__main__":
    main()
