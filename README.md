[Proyecto en PyPi](https://pypi.org/project/ale-uy/)
___
## Modulo [eda.py](): Manipulación de Datos

Las clases `eda.EDA` y `eda.Graphs_eda` son una herramienta para realizar manipulaciones y visualizaciones de datos de manera sencilla y eficiente. Estas clases están diseñadas para facilitar diversas tareas relacionadas con el procesamiento y limpieza de los datos.

### Métodos Disponibles

#### Preprocesamiento de Datos (EDA)

1. `EDA.eliminar_unitarios(df)`: Elimina las variables que tienen un solo valor en un DataFrame.

2. `EDA.eliminar_nulos_si(df, p)`: Elimina las columnas con un porcentaje de valores nulos mayor o igual a `p` en un DataFrame.

3. `EDA.imputar_faltantes(df, metodo="mm")`: Imputa los valores faltantes en un DataFrame utilizando el método de la mediana para variables numéricas y el método de la moda para variables categóricas. También es posible utilizar el método de KNN (K-Nearest Neighbors) para imputar los valores faltantes.

4. `EDA.estandarizar_variables(df, metodo="zscore")`: Estandariza las variables numéricas en un DataFrame utilizando el método "z-score" (estandarización basada en la media y desviación estándar). Tambien estan disponibles otros metodos de estandarizacion 'minmax' y 'robust'

5. `EDA.balancear_datos(df, target)`: Realiza un muestreo aleatorio de los datos para balancear las clases en un problema de clasificación binaria. Esto ayuda a mitigar problemas de desequilibrio de clases en el conjunto de datos.

6. `EDA.mezclar_datos(df)`: Mezcla los datos en el DataFrame de forma aleatoria, lo que puede ser útil para dividir los datos en conjuntos de entrenamiento y prueba.

7. `EDA.estadisticos_numerico(df)`: Genera datos estadísticos de las variables numéricas en el DataFrame.

8. `EDA.convertir_a_numericas(df, target, metodo="ohe")`: Realiza la codificación de variables categóricas utilizando diferentes métodos. Ademas de "ohe" (one-hot-encode) se puede seleccionar "dummy" y "label" (label-encode)

9. `EDA.all_eda(...)`: Pipeline para realizar varios pasos (o todos) de la clase de forma automatica.

#### Visualización de Datos (Graphs_eda)

10. `Graphs_eda.graficos_categoricos(df)`: Crea gráficos de barras horizontales para cada variable categórica en el DataFrame.

11. `Graphs_eda.grafico_histograma(df, x)`: Genera un histograma interactivo para una columna específica del DataFrame.

12. `Graphs_eda.grafico_caja(df, x, y)`: Genera un gráfico de caja interactivo para una variable y en función de otra variable x.

13. `Graphs_eda.grafico_dispersion(df, x, y)`: Genera un gráfico de dispersión interactivo para dos variables x e y.

14. `Graphs_eda.grafico_dendrograma(df)`: Genera un dendrograma que es útil para determinar el valor de k (grupos) para usar con la imputacion knn.

## Modulo [ml.py](): Modelado de Datos

Las clases `ml.ML`, `ml.Graphs_ml` y `ml.Tools` son una herramienta para realizar modelados, manipulación y visualización de datos de manera sencilla y eficiente. Estas clases están diseñadas para facilitar diversas tareas relacionadas con el procesamiento, entrenamiento y evaluación de modelos de aprendizaje automático.

### Modelado de Datos
1. `ML.modelo_lightgbm(...)`: Utiliza LightGBM para predecir la variable objetivo en un DataFrame. Este método admite problemas de clasificación y regresión.

2. `ML.modelo_xgboost(...)`: Utiliza XGBoost para predecir la variable objetivo en un DataFrame. Este método también es adecuado para problemas de clasificación y regresión.

3. `ML.modelo_catboost(...)`: Utiliza CatBoost para predecir la variable objetivo en un DataFrame. Al igual que los métodos anteriores, puede manejar problemas de clasificación y regresión.

> *IMPORTANTE*: si se pasa como parametro ``grid=True`` a cualquiera de estos modelos (ejemplo: **model_catboost(..., grid=True...)**), ahora se realiza una busqueda de hiperparametros **aleatoria** para reducir los tiempos de entrenamiento; ademas podemos pasar ``n_iter=...`` con el numero que deseemos que el modelo pruebe de convinaciones diferentes de parametros (10 es la opcion por defecto).

#### Evaluación de Modelos

5. **Metricas de Clasificación**: Calcula varias métricas de evaluación para un problema de clasificación, como *precisión*, *recall*, *F1-score* y área bajo la curva ROC (*AUC-ROC*).

6. **Metricas de Regresión**: Calcula diversas métricas de evaluación para un problema de regresión, incluyendo el error cuadrático medio (MSE), el coeficiente de determinación (R-cuadrado ajustado), entre otros.

#### Selección de Variables y Clusters

7. `Tools.importancia_variables(...)`: Calcula la importancia de las variables en función de su contribución a la predicción, utiliza Bosque Aleatorio (RandomForest) con validacion cruzada. Utiliza un umbral que determina la importancia mínima requerida para mantener una variable o eliminarla.

8. `Tools.generar_clusters(df)`: Aplica el algoritmo no-supervisado K-Means o DBSCAN a un DataFrame y devuelve una serie con el número de cluster al que pertenece cada observación.

9. `Tools.generar_soft_clusters(df)`: Aplica Gaussian Mixture Models (GMM) al dataframe para generar una tabla con las probabilidades de pertencia de cada observacion al cluster especifico.

10. `Graphs_ml.plot_cluster(df)`: Gráfico de codo y silueta que es escencial para determinar el número de clusters óptimo a utilizar en los métodos de clusters anteriores.

## Modulo [ts.py](): Manipulación de Datos temporales

Las clases `ts.Ts`, `ts.Graphs_ts` y `ts.Profeta` son una poderosa herramienta para realizar modelados, manipulación y visualización de datos temporales. Estas clases están diseñadas para facilitar diversas tareas relacionadas con los datos estadisticos de series temporales, asi como modelado y predicción de los mismo.

### Métodos Disponibles

#### Clase TS
Cada método tiene su funcionalidad específica relacionada con el análisis y la manipulación de series temporales. Puede utilizar estos métodos para realizar diversas tareas en datos de series temporales, incluida la carga de datos, el análisis estadístico, las pruebas de estacionariedad, la descomposición, la diferenciación, la transformación y el modelado SARIMA.

1. `TS.datos_estadisticos(...)`: Este método calcula varias propiedades estadísticas de una serie temporal, como media, mediana, desviación estándar, mínimo, máximo, percentiles, coeficiente de variación, asimetría y curtosis. Devuelve estas estadísticas como un diccionario.
2. `TS.pruebas_raiz_unitaria(...)`: Este método realiza pruebas de raíz unitarias para determinar si una serie temporal es estacionaria. Admite tres pruebas diferentes: Augmented Dickey-Fuller (ADF), Kwiatkowski-Phillips-Schmidt-Shin (KPSS) y Phillips Perron (PP). Devuelve información de diagnóstico y, si es necesario, realiza la diferenciación para hacer que la serie sea estacionaria.
3. `TS.aplicar_descomposicion(...)`: Este método aplica la descomposición estacional a una serie temporal, separándola en tendencia, estacionalidad y residuos. Puede especificar el tipo de descomposición (aditiva o multiplicativa) y el período estacional.
4. `TS.aplicar_diferenciacion(...)`: Este método realiza la diferenciación en una serie temporal para hacerla estacionaria. Puede especificar el número de períodos que va a diferenciar.
5. `TS.aplicar_transformacion(...)`: Este método aplica transformaciones a una serie temporal. Admite tres métodos de transformación: Box-Cox, Yeo-Johnson y logarítmica. Devuelve la serie temporal transformada.
6. `TS.modelo_sarima(...)`: Este método se ajusta a un modelo ARIMA estacional (SARIMA) a una serie temporal. Puede especificar los órdenes de modelo para autorregresivo (AR), diferenciación (d), media móvil (MA), autorregresivo estacional (SAR), diferenciación estacional (D), promedio móvil estacional (SMA) y los períodos estacionales. Devuelve los resultados de la adaptación del modelo SARIMA.

#### Clase Graphs_ts
Estos métodos son útiles para explorar y comprender datos de series temporales, identificar patrones y evaluar supuestos de modelos. Para utilizar estos métodos, debe pasar un DataFrame pandas que contenga datos de series temporales y especificar las columnas y parámetros relevantes.

7. `Graphs_ts.graficar_autocorrelacion(...)`: Este método visualiza la función de autocorrelación (ACF), la función de autocorrelación parcial (PACF) y la ACF estacional de una serie temporal (Sacf y Spacf). Puede especificar el número de retrasos y el nivel de significación de las pruebas.
8. `Graphs_ts.graficar_estacionalidad_tendencia_ruido(...)`:  Este método descompone una serie temporal en su tendencia, estacionalidad y componentes residuales utilizando un modelo aditivo o multiplicativo. A continuación, traza estos componentes junto con la serie temporal original.
9. `Graphs_ts.graficar_diagrama_caja(...)`: Este método genera y muestra diagramas de caja para visualizar datos agrupados por año, mes, día, etc. Puede especificar la columna de tiempo, la columna de valor y la opción de agrupación.
10. `Graphs_ts.graficar_correlograma(...)`: Este método crea y muestra un correlograma (gráfico de autocorrelación) para una serie temporal. Ayuda a identificar correlaciones entre diferentes retrasos en la serie.
11. `Graphs_ts.graficar_profeta(...)`: Este método genera gráficos relacionados con un modelo de Profeta y sus predicciones. Puede elegir visualizar los componentes (tendencia, estacionalidad) o toda la predicción.

#### Clase Profeta:
12. `Profeta.cargar_modelo_prophet(...)`: Este método carga un modelo de Prophet guardado previamente desde un archivo JSON. Puede especificar el nombre del archivo del modelo que se va a cargar.
13. `Profeta.entrenar_modelo(...)`: Este método entrena y ajusta un modelo de Profeta para el pronóstico de series temporales.

## Instalación

Para utilizar las clases `ML`, `EDA`, `Graphs_ml`, `Graphs_eda`, `Tools`, simplemente importa la clase en tu código (primero instalar con pip ``pip install ale-uy``):

```python
from ale_uy.eda import EDA, Graphs_eda
from ale_uy.ml import ML, Tools, Graphs_ml
from ale_uy.ts import TS, Graphs_ts, Profeta
```

## Ejemplo de Uso
Aquí tienes un ejemplo de cómo usar la clase **EDA** y **ML** para realizar un preprocesamiento de datos y entrenar un modelo de LightGBM para un problema de clasificación binaria (IMPORTANTE: Colocar la carpeta **`ale_uy/`** con sus correspondientes archivos **[eda.py]()**, **[ts.py]()** y **[ml.py]()** en la carpeta donde estes trabajando, si es que no instalaste via pip (``pip install ale-uy``)):

```python
# Importar los modulos ml y eda con sus respectivas clases
from ale_uy.ml import ML, Tools, Graphs_ml

from ale_uy.eda import EDA, Graphs_eda

# Cargar los datos en un DataFrame
data = pd.read_csv(...)  # Tu DataFrame con los datos

# Preprocesamiento de datos con la viariable objetivo llamada 'target'
preprocessed_data = EDA.all_eda(data, target='target')

# Entrenar el modelo LightGBM de clasificación y obtener sus metricas
ML.modelo_lightgbm(preprocessed_data, target='target', tipo_problema='clasificacion')

# Si el modelo se adapta a nuestras necesidades, podemos guardarlo simplemente agregando el atributo 'save_model=True'
ML.modelo_lightgbm(preprocessed_data, target='target', tipo_problema='clasificacion', save_model=True)
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
