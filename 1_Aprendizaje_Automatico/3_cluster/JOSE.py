"""
INSTITUTO TECNOLÓGICO SUPERIOR DE ATLIXCO
APRENDIZAJE AUTOMÁTICO
Actividad: K-Means - DBSCAN - PCA

Este script contiene el código completo para las tres partes de la actividad.
Cada sección está claramente marcada para facilitar su integración en un notebook.
"""

# ============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from sklearn.datasets import load_digits, make_blobs, load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Configuración para mejorar la visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PARTE 1: ALGORITMO K-MEANS
# ============================================================================

print("=" * 80)
print("PARTE 1: ALGORITMO K-MEANS")
print("=" * 80)

# ----------------------------------------------------------------------------
# A. Carga y exploración del dataset de dígitos
# ----------------------------------------------------------------------------

print("\n--- A. Carga del Dataset de Dígitos ---\n")

# Cargar el dataset de dígitos manuscritos
data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

print(f"Número de dígitos (clases): {n_digits}")
print(f"Número de muestras: {n_samples}")
print(f"Número de características: {n_features}")
print(f"Dimensiones de los datos: {data.shape}")

# Visualizar algunos ejemplos de dígitos
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Ejemplos de Dígitos del Dataset', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(data[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Dígito: {labels[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# B. Función de evaluación K-Means
# ----------------------------------------------------------------------------

print("\n--- B. Función de Evaluación K-Means ---\n")

def bench_k_means(kmeans, name, data, labels):
    """
    Función para evaluar el rendimiento de KMeans con diferentes métodos de inicialización.
    
    Parameters
    ----------
    kmeans : KMeans instance
        Instancia de KMeans con la inicialización ya configurada.
    name : str
        Nombre de la estrategia de inicialización.
    data : ndarray of shape (n_samples, n_features)
        Los datos a agrupar.
    labels : ndarray of shape (n_samples,)
        Las etiquetas verdaderas para calcular métricas de clustering.
        
    Returns
    -------
    results : list
        Lista con los resultados de las métricas.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Métricas que requieren etiquetas verdaderas y etiquetas predichas
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # Coeficiente de Silhouette (requiere el dataset completo)
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Mostrar los resultados
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))
    
    return results

# ----------------------------------------------------------------------------
# C. Comparación de métodos de inicialización con n_clusters = 10
# ----------------------------------------------------------------------------

print("\n--- C. Comparación de Métodos de Inicialización (n_clusters=10) ---\n")

print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

# K-means++ con 4 inicializaciones
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

# Inicialización aleatoria con 4 inicializaciones
kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

# Inicialización basada en PCA
pca = PCA(n_components=n_digits).fit(data)
kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

print(82 * "_")

# ----------------------------------------------------------------------------
# D. Visualización de K-Means con PCA (2D)
# ----------------------------------------------------------------------------

print("\n--- D. Visualización de K-Means con Reducción PCA ---\n")

# Reducir datos a 2 dimensiones con PCA
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

# Paso del mesh para la frontera de decisión
h = 0.02

# Crear el mesh
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predecir para cada punto en el mesh
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar la frontera de decisión
plt.figure(figsize=(12, 8))
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

# Marcar los centroides con una X blanca
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)

plt.title(
    "K-means clustering en el dataset de dígitos (datos reducidos con PCA)\n"
    "Los centroides están marcados con una cruz blanca",
    fontsize=14,
    fontweight='bold'
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Primera Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.colorbar(label='Cluster')
plt.show()

# ----------------------------------------------------------------------------
# E. MODIFICACIONES: Diferentes números de clusters
# ----------------------------------------------------------------------------

print("\n--- E. MODIFICACIONES: Variación del Número de Clusters ---\n")

# Probar con diferentes números de clusters
n_clusters_list = [8, 10, 12]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparación de K-Means con Diferente Número de Clusters', 
             fontsize=16, fontweight='bold')

for idx, n_clust in enumerate(n_clusters_list):
    print(f"\n--- Evaluando con {n_clust} clusters ---")
    
    # Entrenar K-Means
    kmeans_temp = KMeans(init="k-means++", n_clusters=n_clust, n_init=4, random_state=0)
    kmeans_temp.fit(reduced_data)
    
    # Predecir para el mesh
    Z_temp = kmeans_temp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_temp = Z_temp.reshape(xx.shape)
    
    # Visualizar
    axes[idx].imshow(
        Z_temp,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    axes[idx].plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=1)
    
    centroids_temp = kmeans_temp.cluster_centers_
    axes[idx].scatter(
        centroids_temp[:, 0],
        centroids_temp[:, 1],
        marker="x",
        s=150,
        linewidths=3,
        color="w",
        zorder=10,
    )
    
    axes[idx].set_title(f'n_clusters = {n_clust}\nInertia: {kmeans_temp.inertia_:.0f}')
    axes[idx].set_xlabel('PC1')
    axes[idx].set_ylabel('PC2')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# F. MODIFICACIONES: Diferentes valores de n_init
# ----------------------------------------------------------------------------

print("\n--- F. MODIFICACIONES: Variación de n_init ---\n")

n_init_list = [1, 4, 10, 20]

print("\nComparación con diferentes valores de n_init:")
print(82 * "_")
print("n_init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

for n_init_val in n_init_list:
    kmeans_temp = KMeans(init="k-means++", n_clusters=n_digits, n_init=n_init_val, random_state=0)
    bench_k_means(kmeans=kmeans_temp, name=f"n_init={n_init_val}", data=data, labels=labels)

print(82 * "_")

# ----------------------------------------------------------------------------
# G. Visualización de centroides como imágenes
# ----------------------------------------------------------------------------

print("\n--- G. Visualización de Centroides como Imágenes ---\n")

# Entrenar en datos completos (no reducidos)
kmeans_full = KMeans(init="k-means++", n_clusters=n_digits, n_init=10, random_state=0)
kmeans_full.fit(data)

# Visualizar los centroides
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Centroides de K-Means (reconstruidos como imágenes)', 
             fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(kmeans_full.cluster_centers_[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Cluster {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# PARTE 2: ALGORITMO DBSCAN
# ============================================================================

print("\n" + "=" * 80)
print("PARTE 2: ALGORITMO DBSCAN")
print("=" * 80)

# ----------------------------------------------------------------------------
# A. Generación del dataset sintético
# ----------------------------------------------------------------------------

print("\n--- A. Generación del Dataset Sintético ---\n")

# Crear dataset con 3 centros
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

# Estandarizar los datos
X = StandardScaler().fit_transform(X)

print(f"Número de muestras: {X.shape[0]}")
print(f"Número de características: {X.shape[1]}")
print(f"Centros reales: {len(centers)}")

# Visualizar el dataset antes del clustering
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Dataset Sintético (antes de DBSCAN)', fontsize=14, fontweight='bold')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.grid(True, alpha=0.3)
plt.show()

# ----------------------------------------------------------------------------
# B. Aplicación de DBSCAN con parámetros originales
# ----------------------------------------------------------------------------

print("\n--- B. Aplicación de DBSCAN (eps=0.3, min_samples=10) ---\n")

# Aplicar DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels_dbscan = db.labels_

# Número de clusters (ignorando ruido)
n_clusters_ = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_ = list(labels_dbscan).count(-1)

print(f"Número estimado de clusters: {n_clusters_}")
print(f"Número estimado de puntos de ruido: {n_noise_}")

# Métricas de evaluación
print(f"\nHomogeneidad: {metrics.homogeneity_score(labels_true, labels_dbscan):.3f}")
print(f"Completitud: {metrics.completeness_score(labels_true, labels_dbscan):.3f}")
print(f"V-measure: {metrics.v_measure_score(labels_true, labels_dbscan):.3f}")
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels_dbscan):.3f}")
print(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_true, labels_dbscan):.3f}")
print(f"Coeficiente de Silhouette: {metrics.silhouette_score(X, labels_dbscan):.3f}")

# ----------------------------------------------------------------------------
# C. Visualización de DBSCAN
# ----------------------------------------------------------------------------

print("\n--- C. Visualización de DBSCAN ---\n")

# Identificar puntos núcleo
unique_labels = set(labels_dbscan)
core_samples_mask = np.zeros_like(labels_dbscan, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Crear colores para cada cluster
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(12, 8))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Negro para ruido
        col = [0, 0, 0, 1]

    class_member_mask = labels_dbscan == k

    # Puntos núcleo
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
        label=f'Cluster {k}' if k != -1 else 'Ruido'
    )

    # Puntos de borde
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"DBSCAN - Clusters estimados: {n_clusters_} | Ruido: {n_noise_} puntos",
          fontsize=14, fontweight='bold')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ----------------------------------------------------------------------------
# D. MODIFICACIONES: Variación de eps
# ----------------------------------------------------------------------------

print("\n--- D. MODIFICACIONES: Variación de eps ---\n")

eps_values = [0.05, 0.10, 0.20, 0.3]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle('Comparación de DBSCAN con Diferentes Valores de eps (min_samples=10)', 
             fontsize=16, fontweight='bold')

for idx, eps_val in enumerate(eps_values):
    print(f"\n--- eps = {eps_val} ---")
    
    # Aplicar DBSCAN
    db_temp = DBSCAN(eps=eps_val, min_samples=10).fit(X)
    labels_temp = db_temp.labels_
    
    n_clusters_temp = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
    n_noise_temp = list(labels_temp).count(-1)
    
    print(f"Clusters: {n_clusters_temp} | Ruido: {n_noise_temp} puntos")
    
    # Visualizar
    unique_labels_temp = set(labels_temp)
    core_samples_mask_temp = np.zeros_like(labels_temp, dtype=bool)
    core_samples_mask_temp[db_temp.core_sample_indices_] = True
    
    colors_temp = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels_temp))]
    
    for k, col in zip(unique_labels_temp, colors_temp):
        if k == -1:
            col = [0, 0, 0, 1]
        
        class_member_mask = labels_temp == k
        
        # Puntos núcleo
        xy = X[class_member_mask & core_samples_mask_temp]
        axes[idx].plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col),
                      markeredgecolor="k", markersize=10)
        
        # Puntos de borde
        xy = X[class_member_mask & ~core_samples_mask_temp]
        axes[idx].plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col),
                      markeredgecolor="k", markersize=4)
    
    axes[idx].set_title(f'eps={eps_val} | Clusters: {n_clusters_temp} | Ruido: {n_noise_temp}')
    axes[idx].set_xlabel('Característica 1')
    axes[idx].set_ylabel('Característica 2')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# E. MODIFICACIONES: Variación de min_samples
# ----------------------------------------------------------------------------

print("\n--- E. MODIFICACIONES: Variación de min_samples ---\n")

min_samples_values = [3, 5, 10]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparación de DBSCAN con Diferentes Valores de min_samples (eps=0.3)', 
             fontsize=16, fontweight='bold')

for idx, min_samp in enumerate(min_samples_values):
    print(f"\n--- min_samples = {min_samp} ---")
    
    # Aplicar DBSCAN
    db_temp = DBSCAN(eps=0.3, min_samples=min_samp).fit(X)
    labels_temp = db_temp.labels_
    
    n_clusters_temp = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
    n_noise_temp = list(labels_temp).count(-1)
    
    print(f"Clusters: {n_clusters_temp} | Ruido: {n_noise_temp} puntos")
    
    # Visualizar
    unique_labels_temp = set(labels_temp)
    core_samples_mask_temp = np.zeros_like(labels_temp, dtype=bool)
    core_samples_mask_temp[db_temp.core_sample_indices_] = True
    
    colors_temp = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels_temp))]
    
    for k, col in zip(unique_labels_temp, colors_temp):
        if k == -1:
            col = [0, 0, 0, 1]
        
        class_member_mask = labels_temp == k
        
        # Puntos núcleo
        xy = X[class_member_mask & core_samples_mask_temp]
        axes[idx].plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col),
                      markeredgecolor="k", markersize=10)
        
        # Puntos de borde
        xy = X[class_member_mask & ~core_samples_mask_temp]
        axes[idx].plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col),
                      markeredgecolor="k", markersize=4)
    
    axes[idx].set_title(f'min_samples={min_samp} | Clusters: {n_clusters_temp} | Ruido: {n_noise_temp}')
    axes[idx].set_xlabel('Característica 1')
    axes[idx].set_ylabel('Característica 2')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# PARTE 3: PCA (ANÁLISIS DE COMPONENTES PRINCIPALES)
# ============================================================================

print("\n" + "=" * 80)
print("PARTE 3: PCA (ANÁLISIS DE COMPONENTES PRINCIPALES)")
print("=" * 80)

# ----------------------------------------------------------------------------
# A. Carga del dataset Iris
# ----------------------------------------------------------------------------

print("\n--- A. Carga del Dataset Iris ---\n")

# Cargar dataset Iris
iris = load_iris(as_frame=True)

print("Claves del dataset Iris:")
print(iris.keys())
print(f"\nNúmero de muestras: {iris.data.shape[0]}")
print(f"Número de características: {iris.data.shape[1]}")
print(f"Características: {list(iris.feature_names)}")
print(f"Clases: {list(iris.target_names)}")

# Crear DataFrame con nombres de especies
iris.frame["target"] = iris.target_names[iris.target]

# Visualizar primeras filas
print("\nPrimeras filas del dataset:")
print(iris.frame.head())

# ----------------------------------------------------------------------------
# B. Visualización con pairplot
# ----------------------------------------------------------------------------

print("\n--- B. Visualización con Pairplot ---\n")

# Pairplot para visualizar relaciones entre características
pairplot_fig = sns.pairplot(iris.frame, hue="target", height=2.5)
pairplot_fig.fig.suptitle('Pairplot del Dataset Iris', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# ----------------------------------------------------------------------------
# C. Estandarización de datos
# ----------------------------------------------------------------------------

print("\n--- C. Estandarización de Datos ---\n")

# Separar características y target
X_iris = iris.data.values
y_iris = iris.target.values

# Estandarizar datos
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

print("Datos estandarizados correctamente")
print(f"Media de características después de escalar: {X_iris_scaled.mean(axis=0)}")
print(f"Desviación estándar después de escalar: {X_iris_scaled.std(axis=0)}")

# ----------------------------------------------------------------------------
# D. Aplicación de PCA con 2 componentes
# ----------------------------------------------------------------------------

print("\n--- D. Aplicación de PCA con 2 Componentes ---\n")

# Aplicar PCA con 2 componentes
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_iris_scaled)

print(f"Varianza explicada por cada componente: {pca_2.explained_variance_ratio_}")
print(f"Varianza total explicada: {pca_2.explained_variance_ratio_.sum():.4f} ({pca_2.explained_variance_ratio_.sum()*100:.2f}%)")

# ----------------------------------------------------------------------------
# E. Visualización de PCA 2D
# ----------------------------------------------------------------------------

print("\n--- E. Visualización de PCA 2D ---\n")

# Crear DataFrame para facilitar la visualización
pca_df_2 = pd.DataFrame(
    data=X_pca_2,
    columns=['PC1', 'PC2']
)
pca_df_2['species'] = iris.target_names[y_iris]

# Visualizar
plt.figure(figsize=(12, 8))
colors_iris = ['red', 'green', 'blue']
markers = ['o', 's', '^']

for i, species in enumerate(iris.target_names):
    mask = pca_df_2['species'] == species
    plt.scatter(
        pca_df_2.loc[mask, 'PC1'],
        pca_df_2.loc[mask, 'PC2'],
        c=colors_iris[i],
        label=species,
        marker=markers[i],
        s=100,
        alpha=0.7,
        edgecolors='k'
    )

plt.xlabel(f'PC1 ({pca_2.explained_variance_ratio_[0]*100:.2f}% varianza)', fontsize=12)
plt.ylabel(f'PC2 ({pca_2.explained_variance_ratio_[1]*100:.2f}% varianza)', fontsize=12)
plt.title('PCA del Dataset Iris - Proyección 2D', fontsize=14, fontweight='bold')
plt.legend(title='Especies', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# F. MODIFICACIONES: Varianza explicada
# ----------------------------------------------------------------------------

print("\n--- F. MODIFICACIONES: Análisis de Varianza Explicada ---\n")

# Aplicar PCA con todas las componentes
pca_full = PCA()
pca_full.fit(X_iris_scaled)

# Varianza explicada
variance_explained = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print("Varianza explicada por cada componente:")
for i, (var, cum_var) in enumerate(zip(variance_explained, cumulative_variance)):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%) | Acumulada: {cum_var:.4f} ({cum_var*100:.2f}%)")

# ----------------------------------------------------------------------------
# G. MODIFICACIONES: Scree Plot
# ----------------------------------------------------------------------------

print("\n--- G. MODIFICACIONES: Scree Plot (Gráfica de Codo) ---\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfica de varianza explicada individual
ax1.bar(range(1, len(variance_explained) + 1), variance_explained, 
        alpha=0.7, color='steelblue', edgecolor='black')
ax1.set_xlabel('Componente Principal', fontsize=12)
ax1.set_ylabel('Varianza Explicada', fontsize=12)
ax1.set_title('Varianza Explicada por Componente', fontsize=14, fontweight='bold')
ax1.set_xticks(range(1, len(variance_explained) + 1))
ax1.grid(True, alpha=0.3, axis='y')

# Gráfica de varianza acumulada
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
         marker='o', linestyle='-', linewidth=2, markersize=10, color='coral')
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
ax2.set_xlabel('Número de Componentes', fontsize=12)
ax2.set_ylabel('Varianza Explicada Acumulada', fontsize=12)
ax2.set_title('Scree Plot - Varianza Acumulada', fontsize=14, fontweight='bold')
ax2.set_xticks(range(1, len(cumulative_variance) + 1))
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# H. MODIFICACIONES: PCA con 3 componentes
# ----------------------------------------------------------------------------

print("\n--- H. MODIFICACIONES: PCA con 3 Componentes ---\n")

# Aplicar PCA con 3 componentes
pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_iris_scaled)

print(f"Varianza explicada por cada componente: {pca_3.explained_variance_ratio_}")
print(f"Varianza total explicada: {pca_3.explained_variance_ratio_.sum():.4f} ({pca_3.explained_variance_ratio_.sum()*100:.2f}%)")

# Crear DataFrame para 3 componentes
pca_df_3 = pd.DataFrame(
    data=X_pca_3,
    columns=['PC1', 'PC2', 'PC3']
)
pca_df_3['species'] = iris.target_names[y_iris]

# Visualización 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for i, species in enumerate(iris.target_names):
    mask = pca_df_3['species'] == species
    ax.scatter(
        pca_df_3.loc[mask, 'PC1'],
        pca_df_3.loc[mask, 'PC2'],
        pca_df_3.loc[mask, 'PC3'],
        c=colors_iris[i],
        label=species,
        marker=markers[i],
        s=100,
        alpha=0.7,
        edgecolors='k'
    )

ax.set_xlabel(f'PC1 ({pca_3.explained_variance_ratio_[0]*100:.2f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca_3.explained_variance_ratio_[1]*100:.2f}%)', fontsize=11)
ax.set_zlabel(f'PC3 ({pca_3.explained_variance_ratio_[2]*100:.2f}%)', fontsize=11)
ax.set_title('PCA del Dataset Iris - Proyección 3D', fontsize=14, fontweight='bold')
ax.legend(title='Especies', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualización de pares de componentes en 2D
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Pares de Componentes Principales (3D)', fontsize=16, fontweight='bold')

component_pairs = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC2', 'PC3')]

for idx, (pc_x, pc_y) in enumerate(component_pairs):
    for i, species in enumerate(iris.target_names):
        mask = pca_df_3['species'] == species
        axes[idx].scatter(
            pca_df_3.loc[mask, pc_x],
            pca_df_3.loc[mask, pc_y],
            c=colors_iris[i],
            label=species,
            marker=markers[i],
            s=80,
            alpha=0.7,
            edgecolors='k'
        )
    
    pc_x_idx = int(pc_x[-1]) - 1
    pc_y_idx = int(pc_y[-1]) - 1
    axes[idx].set_xlabel(f'{pc_x} ({pca_3.explained_variance_ratio_[pc_x_idx]*100:.2f}%)', fontsize=11)
    axes[idx].set_ylabel(f'{pc_y} ({pca_3.explained_variance_ratio_[pc_y_idx]*100:.2f}%)', fontsize=11)
    axes[idx].set_title(f'{pc_x} vs {pc_y}')
    axes[idx].legend(title='Especies')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# CONCLUSIONES GENERALES
# ============================================================================

print("\n" + "=" * 80)
print("CONCLUSIONES GENERALES")
print("=" * 80)

print("""
1. K-MEANS:
   - Algoritmo de clustering particional que agrupa datos en K clusters predefinidos.
   - La inicialización K-means++ generalmente produce mejores resultados que la aleatoria.
   - El número de clusters afecta significativamente la calidad del agrupamiento.
   - Mayor n_init puede mejorar la estabilidad pero incrementa el tiempo de cómputo.
   - Los centroides representan los prototipos de cada cluster.

2. DBSCAN:
   - Algoritmo de clustering basado en densidad, no requiere especificar K.
   - Identifica clusters de forma arbitraria y detecta puntos de ruido.
   - eps controla el radio de vecindad: valores muy pequeños generan más ruido.
   - min_samples define la densidad mínima: valores altos requieren regiones más densas.
   - Excelente para detectar outliers y clusters de formas irregulares.

3. PCA:
   - Técnica de reducción de dimensionalidad que preserva la máxima varianza.
   - Permite visualizar datos de alta dimensión en 2D o 3D.
   - Las primeras componentes capturan la mayor parte de la información.
   - El scree plot ayuda a determinar el número óptimo de componentes.
   - Útil para eliminar redundancia y visualizar patrones en los datos.

4. COMPARACIÓN:
   - K-Means: rápido, requiere especificar K, asume clusters esféricos.
   - DBSCAN: robusto a outliers, no requiere K, maneja formas arbitrarias.
   - PCA: no es clustering, sino reducción dimensional para preprocesamiento/visualización.
   
5. APLICACIONES:
   - K-Means: segmentación de clientes, compresión de imágenes.
   - DBSCAN: detección de anomalías, análisis geoespacial.
   - PCA: visualización de datos, reducción de ruido, feature engineering.
""")

print("\n" + "=" * 80)
print("FIN DEL ANÁLISIS")
print("=" * 80)
