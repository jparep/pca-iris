import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Iris dataset
df = load_iris()
X = df.data
y = df.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute Covariance Matric
cov_mx = np.cov(X_scaled, rowvar=False)

# Computer Eiggen values and Eigen vectors
eValues, eVectors = np.linalg.eigh(cov_mx)

# Sort eigen values and eigen vectors
sorted_idx = np.argsort(eValues)[::-1]
sorted_eValues = eValues[sorted_idx]
sorted_eVectors = eVectors[:, sorted_idx]

# Select the top 2 eigen vectors (for 2D projection)
n_components = 2
selected_eVectors = sorted_eVectors[:, :n_components]

# Transform the data into new space
X_pca = X_scaled.dot(selected_eVectors)

# Visualize the result
plt.figure(figsize=(8,6))
for i, target_names in enumerate(df.target_names):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], label=target_names)
plt.title("PCA of Iris dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()