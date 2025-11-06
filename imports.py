"""
Imports comunes para todos los notebooks del proyecto TFG.

"""

# ****************************************************
# Imports principales de librerías estándar
# ****************************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ****************************************************
# Imports de scikit-learn (algoritmos y utilidades)
# ****************************************************
# Clustering
from sklearn.cluster import KMeans, DBSCAN

# Preprocesamiento
from sklearn.preprocessing import StandardScaler

# Métricas de rendimiento
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

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


