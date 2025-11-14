# INSTITUTO TECNOLÓGICO SUPERIOR DE ATLIXCO
## APRENDIZAJE AUTOMÁTICO
### Actividad: K-Means - DBSCAN - PCA

*(Nota: El título "Actividad 2. SVM y K-NN" parece ser un error en el documento original, ya que el contenido describe K-Means, DBSCAN y PCA, que coincide con el nombre del archivo "Actividad 3".)*

## 1. Objetivo

Comprender, implementar y analizar los algoritmos Máquinas de Vectores de Soporte (SVM) y k-Vecinos más Cercanos (K-NN) en problemas de clasificación, evaluando cómo los hiperparámetros y el escalado de características afectan el margen, los vectores de soporte, las fronteras de decisión, el sobreajuste y el desempeño cuantitativo del modelo. *(Nota: Este objetivo también parece corresponder a la "Actividad 2" y no coincide con el contenido de K-Means, DBSCAN y PCA descrito a continuación.)*

## 2. Requerimientos

* Google Colab o Jupyter Notebook 
* Librerías: pandas, numpy, matplotlib, scikit-learn, seaborn 

---

## Parte 1: Algoritmo K-Means

### A. Reproduce el ejemplo oficial

* Ejecutar el código tal como aparece en el enlace. 
* Asegurar que se generen todas las gráficas. 

### B. Documentación requerida. En celdas Markdown, explicar:

1.  **Descripción del dataset de dígitos** 
    * Carga del dataset 
    * Número de imágenes 
    * Interpretación visual 
2.  **Proceso del algoritmo K-Means** 
    * Inicialización (k centroides) 
    * Distancia utilizada 
    * Asignación de clústers 
    * Entrenamiento del modelo 
    * Predicciones 
    * Visualización 
3.  **Resultados obtenidos** 
    * Calidad del agrupamiento 
    * Ejemplo de centroides reconstruidos 

### C. Modificaciones obligatorias. El estudiante deberá:

* Cambiar el número de clústers, por lo menos una vez más (por ejemplo, 8, 10, 12). 
    * Probar diferentes valores de $n\_init$. 
* Analizar cómo cambia la gráfica de los centroides. 
* Agregar una sección final: Conclusiones. 

---

## Parte 2: Algoritmo DBSCAN

### A. Reproduce el ejemplo oficial

* Ejecutar el ejemplo según el enlace. 
* Verificar que se generen los clústers y puntos ruidosos. 

### B. Documentación requerida. Explicar:

1.  **Conceptos fundamentales de DBSCAN** 
    * eps 
    * min\_samples 
    * Núcleos, bordes y ruido 
2.  **Explicación del código** 
    * Generación de dataset sintético 
    * Modelo DBSCAN 
    * Identificación de clústers 
    * Visualización final 
3.  **Interpretación de resultados** 
    * ¿Por qué algunos puntos se clasifican como ruido? 
    * ¿Qué forma tienen los clústers? 

### C. Modificaciones obligatorias. El estudiante deberá:

* Cambiar, el valor de eps (por ejemplo, 0.05, 0.10, 0.20). 
* Cambiar min\_samples entre valores como 3, 5 y 10. 
* Comparar al menos 2 gráficas distintas con conclusiones. 

---

## Parte 3: PCA (Análisis de Componentes Principales)

### A. Reproduce el ejemplo oficial

* Ejecutar el ejemplo del enlace, con PCA aplicado al dataset Iris. 
* Asegurarse de obtener la gráfica 2D con colores por especie. 

### B. Documentación requerida. Explicar:

1.  **Qué es PCA** 
    * Reducción de dimensionalidad 
    * Transformación ortogonal 
    * Componentes principales 
2.  **Explicación del código** 
    * Carga del dataset Iris 
    * Estandarización 
    * Cálculo de componentes 
    * Proyección 2D 
3.  **Interpretación de la gráfica** 
    * ¿Qué tan separables son las especies? 
    * ¿Qué representan PC1 y PC2? 

### C. Modificaciones obligatorias. El estudiante deberá:

* Mostrar la varianza explicada por cada componente. 
* Generar una gráfica de codo (scree plot). 
* Probar PCA con 3 componentes y explicar resultados. 

---

## Rúbrica

| Criterio de evaluación | Descripción del desempeño esperado | Puntuación máxima |
| :--- | :--- | :--- |
| 1. Reproducción correcta del ejemplo K-Means | El estudiante copia, ejecuta y adapta correctamente el código del enlace oficial. Se generan todas las gráficas, los centroides y el agrupamiento sin errores. | 15 |
| 2. Documentación técnica del algoritmo K-Means | Describe el dataset de dígitos, explica el funcionamiento del algoritmo, detalla paso a paso el código y documenta con celdas Markdown claras y completas. | 10 |
| 3. Modificaciones solicitadas en K-Means | Cambia el número de clústers y n\_init, presenta resultados comparativos, explica su impacto y muestra conclusiones específicas. | 5 |
| 4. Reproducción correcta del ejemplo DBSCAN | El código del ejemplo oficial se ejecuta sin errores. Se obtienen gráficas con clústers, ruido y etiquetas correctamente mostradas. | 10 |
| 5. Documentación técnica del algoritmo DBSCAN | Explica conceptos clave: eps, min\_samples, puntos núcleo, borde y ruido. Documenta el código y la interpretación gráfica. | 10 |
| 6. Modificaciones solicitadas en DBSCAN | Cambia eps y min\_samples, genera al menos 3 visualizaciones diferentes y explica claramente cómo afectan la forma de los clústers y los puntos de ruido. | 5 |
| 7. Reproducción correcta del ejemplo PCA | Ejecuta correctamente el ejemplo oficial con PCA aplicado al dataset Iris. Genera la gráfica de proyección 2D correctamente etiquetada. | 10 |
| 8. Documentación técnica del algoritmo PCA | Explica el concepto de reducción de dimensionalidad, componentes principales, cálculo y funcionamiento paso a paso del código. | 10 |
| 9. Modificaciones solicitadas en PCA | Incluye la varianza explicada, scree plot y prueba PCA con 3 componentes. Interpreta claramente los resultados. | 5 |
| 10. Conclusiones generales de la actividad | Presenta conclusiones propias sobre los tres algoritmos, comparando características, ventajas, limitaciones y comportamientos observados. | 5 |
| 11. Presentación, orden y limpieza del notebook | Notebook limpio, organizado, con títulos, comentarios claros, sin celdas de error, gráficas bien presentadas y buena redacción técnica. | 5 |


## Links
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html

https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py

## Codigo base K-Means

```python   
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np

from sklearn.datasets import load_digits

data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
from time import time

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))

    from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

pca = PCA(n_components=n_digits).fit(data)
kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

print(82 * "_")

import matplotlib.pyplot as plt

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
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
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
```

## Code base DBSCAN

```python
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])
plt.show()

import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
print(
    "Adjusted Mutual Information:"
    f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
)
print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()
```



### Codigo base PCA

```python

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
print(iris.keys())

import seaborn as sns

# Rename classes using the iris target names
iris.frame["target"] = iris.target_names[iris.target]
_ = sns.pairplot(iris.frame, hue="target")

import matplotlib.pyplot as plt

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
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
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
```