import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def calculate_correlation(df):
    return df.corr()

def plot_correlation_matrix(corr_matrix, save_path='correlation_matrix.png'):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(save_path)
    plt.show()

def apply_pca(X, explained_variance=0.80):
    pca = PCA(n_components=explained_variance)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_

def plot_pca_variance(explained_variance_ratio_):
    num_components = len(explained_variance_ratio_)
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_components), explained_variance_ratio_)
    plt.plot(np.cumsum(explained_variance_ratio_))
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance')
    plt.show()
