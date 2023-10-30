## Modulo [eda.py](): Manipulación de Datos

Las clases `eda.EDA`, `eda.Graphs_eda` y `eda.Multivariate` son una herramienta para realizar manipulaciones y visualizaciones de datos de manera sencilla y eficiente. Estas clases están diseñadas para facilitar diversas tareas relacionadas con el procesamiento y limpieza de los datos.

### Métodos Disponibles

#### Preprocesamiento de Datos (EDA)

1. `EDA.remove_single_value_columns(df)`: Elimina las variables que tienen un solo valor en un DataFrame.

2. `EDA.remove_missing_if(df, p=0.5)`: Elimina las columnas con un porcentaje de valores nulos mayor o igual a `p` en un DataFrame.

3. `EDA.impute_missing(df, method="mm", n_neighbors=None)`: Imputa los valores faltantes en un DataFrame utilizando el método de la mediana para variables numéricas y el método de la moda para variables categóricas. También es posible utilizar el método de KNN (K-Nearest Neighbors) para imputar los valores faltantes.

4. `EDA.standardize_variables(df, method="zscore")`: Estandariza las variables numéricas en un DataFrame utilizando el método "z-score" (estandarización basada en la media y desviación estándar). Tambien estan disponibles otros metodos de estandarizacion 'minmax' y 'robust'

5. `EDA.balance_data(df, target, oversampling=True)`: Realiza un muestreo aleatorio de los datos para balancear las clases en un problema de clasificación binaria. Esto ayuda a mitigar problemas de desequilibrio de clases en el conjunto de datos.

6. `EDA.shuffle_data(df)`: Mezcla los datos en el DataFrame de forma aleatoria, lo que puede ser útil para dividir los datos en conjuntos de entrenamiento y prueba.

7. `EDA.numeric_statistics(df)`: Genera datos estadísticos de las variables numéricas en el DataFrame.

8. `EDA.convert_to_numeric(df, method="ohe", drop_first=True)`: Realiza la codificación de variables categóricas utilizando diferentes métodos. Ademas de "ohe" (one-hot-encode) se puede seleccionar "dummy" y "label" (label-encode)

9. `EDA.analyze_nulls(df)`: Devuelve el porcentaje de valores nulos en todo el conjunto de datos para cada columna.

10. `EDA.remove_duplicate(df)`: Eliminar filas duplicadas de un DataFrame.

11. `EDA.remove_outliers(df, method='zscore', threshold=3)`: Elimine los valores atípicos de un DataFrame utilizando diferentes métodos. El método para eliminar valores atípicos puede ser 'zscore' (default) o 'iqr'.

12. `EDA.perform_pca(df, n_components='mle')`: Realiza un Análisis de Componentes Principales (PCA). Puedes usar 'mle' para una selección automática o especificar un número entero para un número fijo de componentes.

#### Visualización de Datos (Graphs_eda)

13. `Graphs_eda.categorical_plots(df)`: Crea gráficos de barras horizontales para cada variable categórica en el DataFrame.

14. `Graphs_eda.histogram_plot(df, column)`: Genera un histograma interactivo para una columna específica del DataFrame.

15. `Graphs_eda.box_plot(df)`: Genera un diagrama de caja combinado para variables numéricas en un DataFrame o una sola Serie.

16. `Graphs_eda.scatter_plot(df, column_x, column_y)`: Genera un gráfico de dispersión interactivo para dos variables x e y.

17. `Graphs_eda.hierarchical_clusters_plot(df, method='single', metric='euclidean', save_clusters=False)`: Genera un dendrograma que es útil para determinar el valor de k (grupos) en clusters jerárquicos.

18. `Graphs_eda.correlation_heatmap_plot(df)`: Genera un mapa de calor de correlación para el DataFrame dado.

19. `Graphs_eda.numerical_plot_density(df)`: Genera gráficos de densidad para todas las variables numéricas.

20. `Graphs_eda.pca_elbow_method_plot(df, target_variance=0.95)`: Realiza un Análisis de Componentes Principales (PCA) y utiliza el método del codo para seleccionar el número de componentes. *target_variance (float)* es la varianza acumulativa objetivo.

#### Modelos de Regresion y Clasificacion (Models)

1. `Models.perform_model(df, target, type_model='linear')`: Este método ajusta un tipo especificado de modelo de regresión a los datos proporcionados. Admite modelos de regresión lineal, logística, de Poisson y de regresión lineal robusta. Los resultados de la regresión se imprimen, y se devuelve el modelo. *type_model* = 'linear' (por defecto), 'logit', 'poisson', 'robust'.

## Modulo [ml.py](): Modelado de Datos

Las clases `ml.ML`, `ml.Graphs_ml` y `ml.Tools` son una herramienta para realizar modelados, manipulación y visualización de datos de manera sencilla y eficiente. Estas clases están diseñadas para facilitar diversas tareas relacionadas con el procesamiento, entrenamiento y evaluación de modelos de aprendizaje automático.

### Modelado de Datos
1. `ML.lightgbm_model(...)`: Utiliza LightGBM para predecir la variable objetivo en un DataFrame. Este método admite problemas de clasificación y regresión. Dentro del docstring se pueden ver los parametros que pueden ser personalizados.

2. `ML.xgboost_model(...)`: Utiliza XGBoost para predecir la variable objetivo en un DataFrame. Este método también es adecuado para problemas de clasificación y regresión. Dentro del docstring se pueden ver los parametros que pueden ser personalizados.

3. `ML.catboost_model(...)`: Utiliza CatBoost para predecir la variable objetivo en un DataFrame. Al igual que los métodos anteriores, puede manejar problemas de clasificación y regresión. Dentro del docstring se pueden ver los parametros que pueden ser personalizados.

> *IMPORTANTE*: si se pasa como parametro ``grid=True`` a cualquiera de estos modelos (ejemplo: **model_catboost(..., grid=True...)**), ahora se realiza una busqueda de hiperparametros **aleatoria** para reducir los tiempos de entrenamiento; ademas podemos pasar ``n_iter=...`` con el numero que deseemos que el modelo pruebe de convinaciones diferentes de parametros (10 es la opcion por defecto).

#### Evaluación de Modelos

5. **Metricas de Clasificación**: Calcula varias métricas de evaluación para un problema de clasificación, como *precisión*, *recall*, *F1-score* y área bajo la curva ROC (*AUC-ROC*).

6. **Metricas de Regresión**: Calcula diversas métricas de evaluación para un problema de regresión, incluyendo el error cuadrático medio (MSE), el coeficiente de determinación (R-cuadrado ajustado), entre otros.

#### Selección de Variables y Clusters

7. `Tools.feature_importance(...)`: Calcula la importancia de las variables en función de su contribución a la predicción, utiliza Bosque Aleatorio (RandomForest) con validacion cruzada. Utiliza un umbral que determina la importancia mínima requerida para mantener una variable o eliminarla. Dentro del docstring se pueden ver los parametros que pueden ser personalizados.

8. `Tools.generate_clusters(...)`: Aplica el algoritmo no-supervisado K-Means o DBSCAN a un DataFrame y devuelve una serie con el número de cluster al que pertenece cada observación. Dentro del docstring se pueden ver los parametros que pueden ser personalizados.

9. `Tools.generate_soft_clusters(...)`: Aplica Gaussian Mixture Models (GMM) al dataframe para generar una tabla con las probabilidades de pertencia de cada observacion al cluster especifico. Dentro del docstring se pueden ver los parametros que pueden ser personalizados.

10. `Graphs_ml.plot_cluster(df, random_state=np.random.randint(1, 1000))`: Gráfico de codo y silueta que es escencial para determinar el número de clusters óptimo a utilizar en los métodos de clusters anteriores.

## Modulo [ts.py](): Manipulación de Datos temporales

Las clases `ts.Ts`, `ts.Graphs_ts` y `ts.Propheta` son una poderosa herramienta para realizar modelados, manipulación y visualización de datos temporales. Estas clases están diseñadas para facilitar diversas tareas relacionadas con los datos estadisticos de series temporales, asi como modelado y predicción de los mismo.

### Métodos Disponibles

#### Clase TS
Cada método tiene su funcionalidad específica relacionada con el análisis y la manipulación de series temporales. Puede utilizar estos métodos para realizar diversas tareas en datos de series temporales, incluida la carga de datos, el análisis estadístico, las pruebas de estacionariedad, la descomposición, la diferenciación, la transformación y el modelado SARIMA.

1. `TS.statistical_data(df, target)`: Este método calcula varias propiedades estadísticas de una serie temporal, como media, mediana, desviación estándar, mínimo, máximo, percentiles, coeficiente de variación, asimetría y curtosis. Devuelve estas estadísticas como un diccionario.
2. `TS.unit_root_tests(df, target, test='adf', alpha="5%")`: Este método realiza pruebas de raíz unitarias para determinar si una serie temporal es estacionaria. Admite tres pruebas diferentes: Augmented Dickey-Fuller (ADF), Kwiatkowski-Phillips-Schmidt-Shin (KPSS) y Phillips Perron (PP). Devuelve información de diagnóstico y, si es necesario, realiza la diferenciación para hacer que la serie sea estacionaria.
3. `TS.apply_decomposition(df, target, seasonal_period, model='additive')`: Este método aplica la descomposición estacional a una serie temporal, separándola en tendencia, estacionalidad y residuos. Puede especificar el tipo de descomposición (aditiva o multiplicativa) y el período estacional.
4. `TS.apply_differencing(df, target, periods=1)`: Este método realiza la diferenciación en una serie temporal para hacerla estacionaria. Puede especificar el número de períodos que va a diferenciar.
5. `TS.apply_transformation(df, target, method='box-cox')`: Este método aplica transformaciones a una serie temporal. Admite tres métodos de transformación: Box-Cox, Yeo-Johnson y logarítmica. Devuelve la serie temporal transformada.
6. `TS.sarima_model(df, target, p=0, d=0, q=0, P=0, D=0, Q=0, s=0)`: Este método ajusta un modelo ARIMA a una serie temporal especificando los órdenes de la parte autorregresiva (AR), diferenciación (d) y la media móvil (MA). A su vez puede ajustar un modelo SARIMA modificando los otros cuatros parametros, orden autorregresivo estacional (P), diferenciación estacional (D), promedio móvil estacional (Q) y los períodos estacionales (s). Devuelve los resultados de la adaptación del modelo ARIMA / SARIMA.

#### Clase Graphs_ts
Estos métodos son útiles para explorar y comprender datos de series temporales, identificar patrones y evaluar supuestos de modelos. Para utilizar estos métodos, debe pasar un DataFrame pandas que contenga datos de series temporales y especificar las columnas y parámetros relevantes.

7. `Graphs_ts.plot_autocorrelation(df, value_col, lags=24, alpha=0.05)`: Este método visualiza la función de autocorrelación (ACF), la función de autocorrelación parcial (PACF) y la ACF estacional de una serie temporal (Sacf y Spacf). Puede especificar el número de retrasos y el nivel de significación de las pruebas.
8. `Graphs_ts.plot_seasonality_trend_residuals(df, value_col, period=12, model='additive')`:  Este método descompone una serie temporal en su tendencia, estacionalidad y componentes residuales utilizando un modelo aditivo o multiplicativo. A continuación, traza estos componentes junto con la serie temporal original.
9. `Graphs_ts.plot_box_plot(df, time_col, value_col, group_by='year')`: Este método genera y muestra diagramas de caja para visualizar datos agrupados por año, mes, día, etc. Puede especificar la columna de tiempo, la columna de valor y la opción de agrupación.
10. `Graphs_ts.plot_correlogram(df, value='value', max_lag=10, title='Correlogram Plot')`: Este método crea y muestra un correlograma (gráfico de autocorrelación) para una serie temporal. Ayuda a identificar correlaciones entre diferentes retrasos en la serie.
11. `Graphs_ts.plot_prophet(model, forecast, plot_components=False)`: Este método genera gráficos relacionados con un modelo de Profeta y sus predicciones. Puede elegir visualizar los componentes (tendencia, estacionalidad) o toda la predicción.

#### Clase Propheta:
12. `Propheta.load_prophet_model(model_name='prophet_model')`: Este método carga un modelo de Prophet guardado previamente desde un archivo JSON. Puede especificar el nombre del archivo del modelo que se va a cargar.
13. `Propheta.train_prophet_model(...)`: Este método entrena y ajusta un modelo de Profeta para el pronóstico de series temporales. Dentro del docstring se pueden ver los parametros que pueden ser personalizados.

## Module [dl.py](): Modelos de Rede Neuronales

La clase `dl.DL` es una herramientas que te ayudará a modelar datos con redes neuronales. Está diseñada para facilitar la tarea de crear el modelado y la predicción con los datos que dispongas.

### Métodos Disponibles:

1. `DL.model_ANN(...)`: cree un modelo de red neuronal artificial (ANN) personalizable utilizando scikit-learn. Puede explorar los parámetros personalizables dentro de la cadena de documentación.
2. `DL.model_FNN(...)`: Crea un modelo de red neuronal de avance (FNN) personalizable. Puede explorar los parámetros personalizables dentro de la cadena de documentación.

## Instalación

Colocar la carpeta **`ale_uy/`** con sus correspondientes archivos **[eda.py]()**, **[ts.py]()**, **[ml.py]()** y **[dl.py]()** en el directorio de trabajo, Luego vaya al cmd (linea de comandos) con click izquierdo y tocar abrir cmd aquí, e instale los requisitos con ``pip install -r requirements.txt`` (IMPORTANTE: se recomienda hacerlo en un entorno virtual limpio, para ver cómo hacerlo vaya a [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html))

Para usar las clases `ML`, `EDA`, `Graphs_ml`, `Graphs_eda`, `DL` y `Tools`, simplemente importa la clase en tu código:

```python
from ale_uy.eda import EDA, Graphs_eda
from ale_uy.ml import ML, Tools, Graphs_ml
from ale_uy.ts import TS, Graphs_ts, Propheta
from ale_uy.dl import DL
```

## Ejemplo de Uso
Aquí tienes un ejemplo de cómo usar la clase **EDA** y **ML** para realizar un preprocesamiento de datos y entrenar un modelo de LightGBM para un problema de clasificación binaria:

```python
# Importar los modulos ml y eda con sus respectivas clases
from ale_uy.ml import ML, Tools, Graphs_ml

from ale_uy.eda import EDA, Graphs_eda

# Cargar los datos en un DataFrame
data = pd.read_csv(...)  # Tu DataFrame con los datos

# Preprocesamiento de datos con la viariable objetivo llamada 'target'
preprocessed_data = EDA.perform_full_eda(data, target='target')

# Entrenar el modelo LightGBM de clasificación y obtener sus metricas
ML.lightgbm_model(preprocessed_data, target='target', problem_type='classification')

# Si el modelo se adapta a nuestras necesidades, podemos guardarlo simplemente agregando el atributo 'save_model=True'
ML.lightgbm_model(preprocessed_data, target='target', problem_type='classification', save_model=True)
# Se guardara como "lightgbm.pkl"
```
Para usar el modelo guardado con nuevos datos, usaremos el siguiente codigo
```python
import joblib

# Ruta y nombre del archivo donde se guardó el modelo
model_filename = "nombre_del_archivo.pkl"
# Cargar el modelo
loaded_model = joblib.load(model_filename)
# Ahora puedes utilizar el modelo cargado para hacer predicciones
# Supongamos que tienes un conjunto de datos 'X_test' para hacer predicciones
y_pred = loaded_model.predict(X_test)
```

## Contribución
Si encuentras algún problema o tienes ideas para mejorar estas clases, ¡no dudes en contribuir! Puedes hacerlo enviando pull requests o abriendo issues en el [Repositorio del Proyecto](https://github.com/ale-uy/DataScience).

¡Gracias por tu interés! Espero que sea una herramienta útil para tus proyectos de aprendizaje automático. Si tienes alguna pregunta o necesitas ayuda, no dudes en preguntar. ¡Buena suerte en tus proyectos de ciencia de datos y aprendizaje automático!
