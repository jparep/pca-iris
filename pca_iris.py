import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Iris dataset
df = load_iris()
features = df.data
target = df.target

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Compute Covariance Matric