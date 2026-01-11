# Universitat Oberta de Catalunya  
## Grado en Ingeniería Informática  
### Trabajo Final de Grado (TFG)

---

## Sistema de recomendaciones basado en técnicas de aprendizaje automático para ampliar la exploración de géneros musicales 
**Autor:** Marc Fernández Pereira  
**Bajo supervisión de:** Dra. María Moreno de Castro  
**Área:** Inteligencia Artificial  
**Semestre:** Otoño 2025  

---

Este repositorio contiene el código, análisis y los modelos desarrollados para el Trabajo Final de Grado que trata sobre un sistema de recomendaciones musicales basado en técnicas de aprendizaje automático. Incluye cuatro cuadernos Jupyter que cubren el análisis exploratorio de datos, implementación de algoritmos de *clustering*, técnicas de explicabilidad e incertidumbre, y evaluación comparativa de modelos.

**Cuaderno de exploración y preprocesado de datos:** [`01_exploracion_preprocesado_datos.ipynb`](01_exploracion_preprocesado_datos.ipynb)

Contiene el EDA donde se lleva a cabo el análisis estadístico descriptivo, identificación de correlaciones, análisis de distribuciones (asimetría y curtosis [ver referencia de la memoria 42]), detección de valores atípicos, visualizaciones por género y popularidad, y exploración de similitudes entre géneros. Además, se llevan a cabo tareas de limpieza, selección de variables y estandarización del conjunto de datos Spotify tracks dataset.

**Cuaderno dedicado al *clustering*:** [`02_clustering.ipynb`](02_clustering.ipynb)

Implementación y evaluación de algoritmos de agrupación no supervisada. Se aplican k-means (con optimización de k mediante método del codo y coeficiente de silueta), k-medoids y DBSCAN/OPTICS, tanto sobre datos estandarizados como sobre componentes principales (PCA, 95% de varianza). Se evalúan configuraciones con 𝑘 = 4 y 𝑘 = 7, se analizan visualizaciones de clústeres en espacios 2D y 3D, y se calculan métricas de validación interna (Silhouette, Davies-Bouldin, Calinski-Harabasz).

**Cuaderno de XAI y UQ:** [`03_explicabilidad_incertidumbre.ipynb`](03_explicabilidad_incertidumbre.ipynb)

Aplicación de técnicas de XAI y UQ adaptadas a *clustering*. Permutation Feature Importance (PFI) y SHAP adaptado al *clustering* (SHAP-C) para análisis local y global por clúster. Además, se lleva a cabo implementación de Gaussian Mixture Model (GMM) con 20 componentes (seleccionado mediante BIC) para cuantificar la incertidumbre probabilística en las asignaciones.

**Cuaderno de evaluación:** [`04_evaluacion.ipynb`](04_evaluacion.ipynb)

Comparativa final de todos modelos desarrollados mediante métricas de validación interna (Silhouette, Davies-Bouldin, Calinski-Harabasz).

**Módulo utils**

Funciones auxiliares reutilizables para visualización, cálculo de métricas estadísticas y evaluación de *clustering*, centralizadas en [`utils/functions.py`](utils/functions.py) e importadas mediante [`imports.py`](imports.py).

---

## Instalación

### Requisitos previos

- Python 3.10
- Gestor de paquetes de python *pip*

### Pasos de instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/marcfpereirauoc/TFG
   cd TFG
   ```

2. **Crear un entorno virtual**:
   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual**:
   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Iniciar Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

Una vez iniciado Jupyter Notebook, podrás abrir y ejecutar los cuadernos en el orden indicado:
1. `01_exploracion_preprocesado_datos.ipynb`
2. `02_clustering.ipynb`
3. `03_explicabilidad_incertidumbre.ipynb`
4. `04_evaluacion.ipynb`

### Notas

- Los archivos `.pkl` (modelos y métricas) no están incluidos en el repositorio. Estos archivos se generan automáticamente al ejecutar los cuadernos y se guardan para poder reutilizarlos en cuadernos posteriores sin necesidad de reentrenar los modelos Esto mejora  la eficiencia computacional del proyecto al no tener que reentrenar los modelos.
