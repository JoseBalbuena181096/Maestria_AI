# Actividad 1. Regresión Lineal Simple con el conjunto de datos "Salary Data"

## 1. Objetivo
[cite_start]Aplicar un modelo de regresión lineal supervisada para predecir el salario de una persona en función de sus años de experiencia[cite: 4].

[cite_start]El estudiante analizará el conjunto de datos, dividirá la información en subconjuntos de entrenamiento y prueba, ajustará un modelo lineal, evaluará su desempeño y visualizará los resultados[cite: 5].

---

## 2. Descripción del conjunto de datos
Descargar el conjunto de datos de `Salary_Data.csv` el cual se encuentra disponible en:
[cite_start]https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression?resource=download [cite: 7]

[cite_start]Dicho conjunto contiene dos columnas y un total de 30 registros[cite: 8]:

| Variable | Descripción |
| :--- | :--- |
| Years Experience | [cite_start]Años de experiencia laboral del empleado. [cite: 9] |
| Salary | [cite_start]Salario anual en dólares. [cite: 9] |

---

## 3. Requerimientos
* [cite_start]Google Colab o Jupyter Notebook [cite: 11]
* [cite_start]Librerías: pandas, numpy, matplotlib, scikit-learn [cite: 12]

---

## 4. Carga y exploración del conjunto de datos
* [cite_start]Importa las librerías necesarias para el análisis (Pandas, NumPy, Matplotlib y Scikit-Learn)[cite: 14].
* [cite_start]Carga el archivo del conjunto de datos `Salary_Data.csv`[cite: 15].
* [cite_start]Visualiza las primeras filas del conjunto para conocer la estructura de los datos[cite: 16].
* [cite_start]Verifica que no existan valores nulos o inconsistencias[cite: 17].

---

## 5. Análisis exploratorio
* [cite_start]Realiza una gráfica de dispersión (scatter plot) que muestre la relación entre los años de experiencia y el salario[cite: 19].
* [cite_start]Describe si a simple vista existe una relación lineal entre ambas variables[cite: 21].
* [cite_start]Interpreta el comportamiento general de los datos[cite: 24].

---

## 6. Preparación de los datos
* [cite_start]Identifica las variables[cite: 26]:
    * [cite_start]Variable independiente (X) [cite: 27]
    * [cite_start]Variable dependiente (y) [cite: 28]
* [cite_start]Divide el conjunto de datos en dos subconjuntos[cite: 29]:
    * [cite_start]Entrenamiento: 80% de los datos[cite: 30].
    * [cite_start]Prueba: 20% de los datos[cite: 31].
* [cite_start]Recomendación: utilizar la función `train_test_split()` para dividir los datos[cite: 32].
* [cite_start]Explica por qué se realiza esta separación y cuál es su importancia para el entrenamiento supervisado[cite: 33].

---

## 7. Entrenamiento del modelo
* [cite_start]Crea un modelo de Regresión Lineal Simple[cite: 35].
* [cite_start]Ajusta el modelo utilizando los datos de entrenamiento[cite: 36].
* [cite_start]Registra el valor del coeficiente (pendiente) y la intersección obtenidos[cite: 37].
* [cite_start]Interpreta lo que representa cada uno en el contexto del problema (por ejemplo, cómo cambia el salario por cada año adicional de experiencia)[cite: 38, 39].

---

## 8. Pruebas y validación
* [cite_start]Utiliza el modelo entrenado para realizar predicciones sobre el conjunto de prueba[cite: 41].
* [cite_start]Compara los valores reales de salario con los valores predichos en una tabla[cite: 42].
* [cite_start]Realiza una gráfica donde se muestren los puntos reales y la línea de regresión obtenida[cite: 43].
* [cite_start]Analiza visualmente si el modelo se ajusta adecuadamente a los datos[cite: 44].

---

## 9. Evaluación del desempeño
* [cite_start]Calcula las métricas de evaluación más comunes[cite: 47]:
    * [cite_start]MAE (Error Absoluto Medio) [cite: 48]
    * [cite_start]MSE (Error Cuadrático Medio) [cite: 49]
* [cite_start]Interpreta los valores obtenidos e indica si el modelo presenta un buen ajuste o si muestra errores significativos[cite: 50, 51].

---

## 10. Predicción de un nuevo caso
* [cite_start]Realiza la estimación del salario para un nuevo valor de entrada (por ejemplo, una persona con 7.5 años de experiencia)[cite: 53].

---

## 11. Análisis y conclusiones
* [cite_start]Resume los resultados obtenidos durante el proceso[cite: 57].
* [cite_start]Explica qué tan útil es este modelo en la práctica[cite: 58].
* [cite_start]Reflexiona sobre las limitaciones de usar un modelo lineal en comparación con otros modelos de aprendizaje automático[cite: 59].

---

## 12. Extensión y formato
* [cite_start]Extensión máxima de la actividad: 15 páginas[cite: 61].

---

## 13. Rúbrica

| Criterio de evaluación | Descripción del desempeño esperado | Puntuación máxima |
| :--- | :--- | :--- |
| 1. Introducción y objetivo | Se explica claramente el propósito de la actividad y el concepto general de regresión lineal. | [cite_start]5 [cite: 63] |
| 2. Carga y descripción del conjunto de datos | Se identifica correctamente el dataset utilizado | [cite_start]15 [cite: 63] |
| 3. División del conjunto de datos | Se describe y aplica correctamente la función `train_test_split()` indicando la proporción utilizada y su justificación. | [cite_start]15 [cite: 63] |
| 4. Entrenamiento del modelo | Se explica el proceso de ajuste del modelo de regresión lineal, indicando pendiente e intersección. | [cite_start]15 [cite: 63] |
| 5. Validación del modelo | Se muestran los resultados obtenidos al comparar datos reales vs predichos, con interpretación adecuada. | [cite_start]10 [cite: 63] |
| 6. Evaluación del modelo | Se calculan e interpretan correctamente las métricas (MAE, MSE). | [cite_start]10 [cite: 63] |
| 7. Predicción de nuevos valores | Se realiza al menos una predicción con un valor nuevo y se interpreta el resultado. | [cite_start]10 [cite: 63] |
| 8. Análisis y conclusiones | Reflexión sobre los resultados, el desempeño del modelo y las posibles mejoras. | [cite_start]10 [cite: 63] |
| 9. Entrega del trabajo | Se entrega el archivo .ipynb (Jupyter Notebook o colab) correctamente ejecutado y el PDF generado con los resultados completos. | [cite_start]10 [cite: 63] |