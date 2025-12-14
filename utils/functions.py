import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import sklearn.metrics as metrics

# Paleta fija con colores diferenciados (hasta 10 clústeres).
palette_colors = [
    '#1f77b4',  # azul
    '#ff7f0e',  # naranja
    '#2ca02c',  # verde
    '#d62728',  # rojo
    '#9467bd',  # morado
    '#8c564b',  # marrón
    '#17becf',  # cian
    '#7f7f7f',  # gris
    '#bcbd22',  # oliva
    '#000000'   # negro
]


def add_histogram_with_stats(ax, df, variable, kde=True, bw_adjust=1.5, bins=40, show_legend=False, show_title=True, grid_alpha=0.4):
    """
    Función auxiliar que añade un histograma con KDE y líneas estadísticas a un eje dado.
    
    :param ax: eje de matplotlib donde se dibujará el histograma
    :param df: DataFrame con los datos
    :param variable: nombre de la variable a graficar
    :param kde: activar/desactivar la curva KDE
    :param bw_adjust: ajuste del ancho de banda para la densidad kernel
    :param bins: número de bins para el histograma
    :param show_legend: si True, muestra la leyenda en el eje individual
    :param show_title: si True, muestra el título en el eje
    :param grid_alpha: transparencia del grid (por defecto 0.4)
    :return: None
    """
    # Histograma con KDE
    sns.histplot(data=df, x=variable, bins=bins, kde=kde, ax=ax, kde_kws=dict(bw_adjust=bw_adjust))
    
    # Calcular estadísticas
    q1 = df[variable].quantile(0.25)
    q2 = df[variable].quantile(0.5)  # Mediana
    q3 = df[variable].quantile(0.75)
    mean = df[variable].mean()
    std = df[variable].std()
    
    # Añadir líneas verticales para las métricas
    ax.axvline(q1, color='#d62728', linestyle='--', alpha=0.7, linewidth=1.5, label='Q1 - 25%')
    ax.axvline(q2, color='#2ca02c', linestyle='--', alpha=0.7, linewidth=1.5, label='Mediana - 50%')
    ax.axvline(q3, color='#ff7f0e', linestyle='--', alpha=0.7, linewidth=1.5, label='Q3 - 75%')
    ax.axvline(mean, color='#9467bd', linestyle='-', linewidth=2, alpha=0.8, label='Media')
    ax.axvline(mean - std, color='#8c564b', linestyle=':', alpha=0.7, linewidth=1.5, label='Media ± σ')
    ax.axvline(mean + std, color='#8c564b', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # Configurar título y leyenda
    if show_title:
        ax.set_title(f'Distribución de {variable}')
    ax.grid(alpha=grid_alpha)
    
    if show_legend:
        ax.legend()


def plot_distribution(df, variable, kde = True, bw_adjust=1.5, bins=40):
    """
    Función para graficar la distribución de una variable numérica y un boxplot para identificar la presencia de outliers.
    Muestra también estadísticas descriptivas: cuartiles (Q1, mediana, Q3), media y desviación estándar.
    
    :param df: conjunto de datos numéricos
    :param variable: nombre de la variable a graficar
    :param kde: activar/desactivar la curva KDE
    :param bins: número de bins para el histograma
    :param bw_adjust: ajuste del ancho de banda para la densidad kernel
    :return: None
    """
    # Crear una figura con dos subplots en horizontal: histograma (izquierda, 75% del ancho) y boxplot (derecha, 25% del ancho)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), width_ratios=[3, 1])

    # Histograma a la izquierda usando la función auxiliar
    add_histogram_with_stats(ax1, df, variable, kde=kde, bw_adjust=bw_adjust, bins=bins, show_legend=True, show_title=True)

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

    kurtosis = (1 / n) * np.sum(((df[variable] - mean) / std) ** 4) - 3

    return kurtosis

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

def plot_numeric_hist_grid(df, cols=3, kde=True, bw_adjust=1.5, bins=30):
    """
    Este método permite crear un cuadro resumen con histogramas para todas las variables numéricas del conjunto de datos.
    Incluye KDE, métricas estadísticas (Q1, Q3, Media) y una leyenda general.

    :param df: dataFrame
    :param cols: número de columnas en el grid (por defecto 3)
    :param kde: activar/desactivar la curva KDE
    :param bw_adjust: ajuste del ancho de banda para la densidad kernel (bw_adjust=1.5 por defecto)
    :param bins: número de bins para cada histograma (bins=30 por defecto)
    :return: None
    """
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] == 0: raise ValueError("El DataFrame no contiene columnas numéricas")

    vars = list(num_df.columns)
    n_vars = len(vars)
    n_cols = max(1, int(cols))
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4.5*n_rows))
    axs = np.array(axs).flatten()

    # Colores y estilos para las líneas estadísticas (Q1, Mediana, Q3, Media, Media ± σ (std))
    stat_names = {
        'Q1': ('#d62728', '--', 'Q1 - 25%'),
        'Mediana': ('#2ca02c', '--', 'Mediana'),
        'Q3': ('#ff7f0e', '--', 'Q3 - 75%'),
        'Media': ('#9467bd', '-', 'Media'),
        'Media ± σ': ('#8c564b', ':', 'Media ± σ')
    }

    # Crear las líneas solo una vez para la leyenda
    legend_handles = []
    for stat_name, (color, linestyle, label) in stat_names.items():
        line = plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2 if stat_name == 'Media' else 1.5, alpha=0.8 if stat_name == 'Media' else 0.7, label=label)
        legend_handles.append(line)

    for i, var in enumerate(vars):
        ax = axs[i]
        
        # Usar la función auxiliar para crear el histograma con estadísticas
        add_histogram_with_stats(ax, num_df, var, kde=kde, bw_adjust=bw_adjust, bins=bins, show_legend=False, show_title=False, grid_alpha=0.3)
        
        # Obtenemos los coeficientes de asimetría y curtosis
        skewness = get_skewness_coeficient(num_df, var)
        kurtosis = get_kurtosis_coeficient(num_df, var)
        
        # Título personalizado con los coeficientes de asimetría y curtosis
        ax.set_title(f'{var}\n(Skew: {skewness:.3f}, Kurt: {kurtosis:.3f})', fontsize=14, fontweight='bold')
        ax.set_ylabel("Count")
    
    # Eliminamos los ejes vacíos si hay un número impar de variables
    for j in range(n_vars, len(axs)):
        fig.delaxes(axs[j])

    # Leyenda general en la parte inferior de la figura
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, frameon=True, fontsize=14, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    plt.show()

def compute_cluster_metrics(features, labels, silhouette_sample=10000):
    """
    Calcula Silhouette, Davies-Bouldin y Calinski-Harabasz para un agrupamiento dado.

    :param features: array o DataFrame con las características utilizadas en el clustering.
    :param labels: etiquetas asignadas a cada observación.
    :param silhouette_sample: tamaño máximo de muestra para la Silhouette (para datasets grandes).

    :return: diccionario con las tres métricas.
    """

    silhouette = metrics.silhouette_score(features, labels, sample_size=silhouette_sample, random_state=42)
    davies_bouldin = metrics.davies_bouldin_score(features, labels)
    calinski_harabasz = metrics.calinski_harabasz_score(features, labels)

    return {
        "Silhouette": round(silhouette, 2),
        "Davies_Bouldin": round(davies_bouldin, 2),
        "Calinski_Harabasz": round(calinski_harabasz, 2),
    }

