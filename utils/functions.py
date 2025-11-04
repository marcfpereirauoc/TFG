import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_distribution(df, variable, kde = True, bw_adjust=1.5, bins=40):
    """
    Función para graficar la distribución de una variable numérica y un boxplot para identificar la presencia de outliers.
    Muestra también estadísticas descriptivas: cuartiles (Q1, mediana, Q3), media y desviación estándar.
    
    :param df: conjunto de datos numéricos
    :param variable: nombre de la variable a graficar
    :param bins: número de bins para el histograma
    :param bw_adjust: ajuste del ancho de banda para la densidad kernel
    :return: None
    """
    # Crear una figura con dos subplots en horizontal: histograma (izquierda, 75% del ancho) y boxplot (derecha, 25% del ancho)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), width_ratios=[3, 1])
    
    sns.set_style("whitegrid")

    # Histograma a la izquierda
    sns.histplot(data=df, x=variable, bins=bins, kde=kde, ax=ax1, kde_kws=dict(bw_adjust=bw_adjust))
    ax1.set_title(f'Distribución de {variable}')
    ax1.grid(alpha=0.4)
    
    # Obtener cuartiles, media y desviación estándar
    q1 = df[variable].quantile(0.25)
    q2 = df[variable].quantile(0.5)
    q3 = df[variable].quantile(0.75)
    mean = df[variable].mean()
    std = df[variable].std()
    
    # Añadir valores a los ejes
    ax1.axvline(q1, color='#d62728', linestyle='--',  alpha=0.7, label='Q1 - 25%')
    ax1.axvline(q2, color='#2ca02c', linestyle='--', alpha=0.7, label='Mediana - 50%')
    ax1.axvline(q3, color='#1f77b4', linestyle='--', alpha=0.7, label='Q3 - 75%')
    ax1.axvline(mean, color='#9467bd', linestyle='-', linewidth=2, alpha=0.8, label='Media')
    ax1.axvline(mean - std, color='#8c564b', linestyle=':', alpha=0.7, label='Media ± σ')
    ax1.axvline(mean + std, color='#8c564b', linestyle=':', alpha=0.7)
    
    # Añadir leyenda para identificar los valores de los ejes
    ax1.legend()

    # Boxplot a la derecha
    sns.boxplot(data=df, y=variable, ax=ax2)
    ax2.set_title(f'Boxplot de {variable}')
    ax2.grid(alpha=0.4)

    fig.tight_layout()
    plt.show()

def get_skewness_coeficient(df, variable):
    """
    Calcula el coeficiente de asimetría (skewness) de una distribución usando el método basado en momentos (Fisher–Pearson).
    # Referencia (Fisher–Pearson sample skewness): https://en.wikipedia.org/wiki/Skewness#Sample_skewness

    :param distribution :  Lista o array con los valores de la variable numérica.
    :param variable: nombre de la variable a calcular el coeficiente de asimetría
    :return: skewness : float
    Valor del coeficiente de asimetría:
        > 0  -> sesgo positivo (cola derecha)
        < 0  -> sesgo negativo (cola izquierda)
        = 0  -> distribución simétrica
    """
    # Validar que la variable exista y sea numérica
    if variable not in df.columns:
        raise KeyError(f"La variable '{variable}' no existe en el DataFrame")
    if not np.issubdtype(df[variable].dtype, np.number):
        raise ValueError(f"La variable {variable} no es numérica")
    
    # Calcular el número de observaciones, la media y la desviación estándar
    n = len(df[variable])
    mean_value = np.mean(df[variable])
    std_value = np.std(df[variable])
    # Formula para calcular el coeficiente de asimetría
    skewness_value = (n / ((n - 1) * (n - 2))) * (np.sum(((df[variable] - mean_value) / std_value) ** 3))

    return skewness_value

def get_kurtosis_coeficient(df, variable):
    """
    Calcula el coeficiente de curtosis (kurtosis) de una distribución usando el método basado en momentos (Fisher–Pearson).
    Fórmula extraída de: https://en.wikipedia.org/wiki/Kurtosis 
    
    :param distribution :  Lista o array con los valores de la variable numérica.
    :param variable: nombre de la variable a calcular el coeficiente de curtosis
    :return: kurtosis : valor del coeficiente de curtosis
    Valor del coeficiente de curtosis:
    # Validar que la variable exista y sea numérica
    """
    # Validar que la variable exista y sea numérica
    if variable not in df.columns:
        raise KeyError(f"La variable '{variable}' no existe en el DataFrame")
    if not np.issubdtype(df[variable].dtype, np.number):
        raise ValueError(f"La variable {variable} no es numérica")
    n = len(df[variable])
    mean = np.mean(df[variable])
    std = np.std(df[variable])

    kurtosis = (1 / n) * sum(((df[variable] - mean) / std) ** 4) - 3

    return kurtosis


import matplotlib.pyplot as plt
import seaborn as sns
import math

def visualize_discrete_features(df, variables):
    """
    Visualiza una o varias variables discretas/categóricas en subplots usando gráficos de líneas de frecuencia ordenados por índice de categoría.

    - Acepta una cadena (una sola variable) o una lista de nombres de columnas.
    - Organiza automáticamente los subplots en un grid de 2 columnas por fila.

    :param df: Conjunto de datos con las variables discretas/categóricas.
    :param variables: nombre de columna (str) o lista de nombres (list[str]) a visualizar.
    :return: None
    """
    # Si se pasa una sola variable, se convierte en lista
    if isinstance(variables, str):
        variables = [variables]

    # Se comprueba que las variables existan en el conjunto de datos
    for var in variables:
        if var not in df.columns:
            raise KeyError(f"La variable '{var}' no existe en el DataFrame")

    n_vars = len(variables)

    # Se calculan filas y columnas (máximo 2 columnas por fila)
    n_cols = 2
    n_rows = math.ceil(n_vars / n_cols)

    # Estilo del gráfico (figura y ejes)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axs = axs.flatten()

    # Se generan los gráficos de líneas para cada variable
    for i in range(n_vars):
        var = variables[i]
        counts = df[var].value_counts().sort_index()

        axs[i].plot(counts.index, counts.values, color='steelblue')
        axs[i].set_title(f'Distribución de {var} (frecuencia)', fontsize=12, fontweight='bold')
        axs[i].set_xlabel(var, fontsize=10)
        axs[i].set_ylabel('Frecuencia', fontsize=10)
        axs[i].grid(alpha=0.4)

    # Eliminar ejes vacíos si hay un número impar de variables
    for j in range(n_vars, len(axs)):
        fig.delaxs(axs[j])

    # Ajustar espacios
    fig.tight_layout(w_pad=3, h_pad=3)
    plt.show()

def plot_numeric_hist_grid(df, cols=4):
    """
    Crea un cuadro resumen con histogramas para todas las variables numéricas del DataFrame.

    :param df: DataFrame de pandas
    :param bins: número de bins para cada histograma
    :param cols: número de columnas en el grid
    :param kde: si True, superpone curva KDE
    :param sharex: compartir eje X entre subplots
    :param sharey: compartir eje Y entre subplots
    :return: None
    """
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] == 0:
        raise ValueError("El DataFrame no contiene columnas numéricas")

    variables = list(num_df.columns)
    n_vars = len(variables)
    n_cols = max(1, int(cols))
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axs = np.array(axs).flatten()

    for i, var in enumerate(variables):
        ax = axs[i]
        sns.histplot(data=num_df, x=var, bins=30, ax=ax)
        ax.set_title(var)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
    
    for j in range(n_vars, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.show()
