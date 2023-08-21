## Modulo eda_vx.py: Manipulación de Datos

Las clases `eda_vx.EDA` y `eda_vx.Graph` son una herramienta poderosa para realizar manipulaciones y visualizaciones de datos de manera sencilla y eficiente. Estas clases están diseñadas para facilitar diversas tareas relacionadas con el procesamiento y limpieza de los datos.

### Métodos Disponibles

#### Preprocesamiento de Datos (EDA)

1. `eliminar_unitarios(df)`: Elimina las variables que tienen un solo valor en un DataFrame.

2. `eliminar_nulos_si(df, p)`: Elimina las columnas con un porcentaje de valores nulos mayor o igual a `p` en un DataFrame.

3. `imputar_faltantes(df, metodo="mm")`: Imputa los valores faltantes en un DataFrame utilizando el método de la mediana para variables numéricas y el método de la moda para variables categóricas. También es posible utilizar el método de KNN (K-Nearest Neighbors) para imputar los valores faltantes.

4. `estandarizar_variables(df, metodo="zscore")`: Estandariza las variables numéricas en un DataFrame utilizando el método "z-score" (estandarización basada en la media y desviación estándar). Tambien estan disponibles otros metodos de estandarizacion 'minmax' y 'robust'

5. `balancear_datos(df, target)`: Realiza un muestreo aleatorio de los datos para balancear las clases en un problema de clasificación binaria. Esto ayuda a mitigar problemas de desequilibrio de clases en el conjunto de datos.

6. `mezclar_datos(df)`: Mezcla los datos en el DataFrame de forma aleatoria, lo que puede ser útil para dividir los datos en conjuntos de entrenamiento y prueba.

7. `estadisticos_numerico(df)`: Genera datos estadísticos de las variables numéricas en el DataFrame.

8. `convertir_a_numericas(df, target, metodo="ohe")`: Realiza la codificación de variables categóricas utilizando diferentes métodos. Ademas de "ohe" (one-hot-encode) se puede seleccionar "dummy" y "label" (label-encode)

9. `all_eda(...)`: Pipeline para realizar varios pasos (o todos) de la clase de forma automatica.

#### Visualización de Datos (Graph)

10. `graficos_categoricos(df)`: Crea gráficos de barras horizontales para cada variable categórica en el DataFrame.

11. `grafico_histograma(df, x)`: Genera un histograma interactivo para una columna específica del DataFrame.

12. `grafico_caja(df, x, y)`: Genera un gráfico de caja interactivo para una variable y en función de otra variable x.

13. `grafico_dispersion(df, x, y)`: Genera un gráfico de dispersión interactivo para dos variables x e y.

14. `grafico_dendrograma(df)`: Genera un dendrograma que es útil para determinar el valor de k (grupos) para usar con la imputacion knn.

## Modulo ml_vx.py: Modelado de Datos

La clase `ml_vx.ML` es una herramienta poderosa para realizar modelados y visualizacion de datos de manera sencilla y eficiente. Esta clase están diseñadas para facilitar diversas tareas relacionadas con el procesamiento, entrenamiento y evaluación de modelos de aprendizaje automático.

### Modelado de Datos
1. `modelo_lightgbm(...)`: Utiliza LightGBM para predecir la variable objetivo en un DataFrame. Este método admite problemas de clasificación y regresión.

2. `modelo_xgboost(...)`: Utiliza XGBoost para predecir la variable objetivo en un DataFrame. Este método también es adecuado para problemas de clasificación y regresión.

3. `modelo_catboost(...)`: Utiliza CatBoost para predecir la variable objetivo en un DataFrame. Al igual que los métodos anteriores, puede manejar problemas de clasificación y regresión.

> *IMPORTANTE*: si se pasa como parametro ``grid=True`` a cualquiera de estos modelos (ejemplo: **model_catboost(..., grid=True...)**), ahora se realiza una busqueda de hiperparametros **aleatoria** para reducir los tiempos de entrenamiento; ademas podemos pasar ``n_iter=...`` con el numero que deseemos que el modelo pruebe de convinaciones diferentes de parametros (10 es la opcion por defecto).

#### Evaluación de Modelos

5. **Metricas de Clasificación**: Calcula varias métricas de evaluación para un problema de clasificación, como *precisión*, *recall*, *F1-score* y área bajo la curva ROC (*AUC-ROC*).

6. **Metricas de Regresión**: Calcula diversas métricas de evaluación para un problema de regresión, incluyendo el error cuadrático medio (MSE), el coeficiente de determinación (R-cuadrado ajustado), entre otros.

#### Selección de Variables

7. `importancia_variables(...)`: Calcula la importancia de las variables en función de su contribución a la predicción, utiliza Bosque Aleatorio (RandomForest) con validacion cruzada. Utiliza un umbral que determina la importancia mínima requerida para mantener una variable o eliminarla.

8. `generar_clusters(df)`: Aplica el algoritmo no-supervisado K-Means o DBSCAN a un DataFrame y devuelve una serie con el número de cluster al que pertenece cada observación.

### Instalación

Para utilizar las clases `ML`, `EDA`, `Graph`, `Tools`, simplemente importa la clase en tu código (copia la carpeta vx en el directorio en que estes trabajando):

```python
from vx.ml_vx import ML, Tools
from vx.eda_vx import EDA, Graph
```
Recuerda asegurarte de tener las librerías necesarias instaladas en tu entorno (En anaconda solo necesitaras instalar *Lightgbm*, *Xgboost* y *Catboost*). Puedes instalar las librerías requeridas utilizando el siguiente comando en command-line (cmd, windows):

```bash
pip install -r requirements.txt
```

## Ejemplo de Uso
Aquí tienes un ejemplo de cómo usar la clase **EDA** y **ML** para realizar un preprocesamiento de datos y entrenar un modelo de LightGBM para un problema de clasificación binaria (IMPORTANTE: Colocar los archivos **eda_vx.py** y **ml_vx.py** en la carpeta donde estes trabajando):

```python
# Importar la clase ML
from vx.ml_vx import ML, Tools
from vx.eda_vx import EDA, Graph

# Cargar los datos en un DataFrame
data = pd.read_csv(...)  # Tu DataFrame con los datos

# Preprocesamiento de datos
preprocessed_data = EDA.all_eda(data, target='target')

# Entrenar el modelo LightGBM y obtener sus metricas
ML.modelo_lightgbm(preprocessed_data, target='target', tipo_problema='clasificacion')

# Si el modelo se adapta a nuestras necesidades, podemos guardarlo simplemente agregando el atributo save_model=True
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
Si encuentras algún problema o tienes ideas para mejorar estas clases, ¡no dudes en contribuir! Puedes hacerlo enviando pull requests o abriendo issues en el repositorio del proyecto.

¡Gracias por tu interés! Espero que sea una herramienta útil para tus proyectos de aprendizaje automático. Si tienes alguna pregunta o necesitas ayuda, no dudes en preguntar. ¡Buena suerte en tus proyectos de ciencia de datos y aprendizaje automático!
