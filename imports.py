"""
Imports comunes para todos los notebooks del proyecto TFG.

"""
import joblib

# ****************************************************
# Imports principales de librerías estándar
# ****************************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import display

# ****************************************************
# Imports de scikit-learn (algoritmos y utilidades)
# ****************************************************
# Clustering
from sklearn.cluster import KMeans, DBSCAN

# Preprocesamiento
from sklearn.preprocessing import StandardScaler
# Reducción de dimensionalidad
from sklearn.decomposition import PCA

# Métricas de rendimiento
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances_argmin_min
)

from sklearn.decomposition import PCA

# ****************************************************
# Imports de funciones personalizadas del proyecto
# ****************************************************
from utils.functions import (
    plot_distribution,
    get_skewness_coeficient,
    get_kurtosis_coeficient,
    visualize_discrete_features,
    plot_numeric_hist_grid
)

# ****************************************************
# Configuración global del proyecto
# ****************************************************

# Configuración de estilo para gráficos
sns.set_style("whitegrid")


