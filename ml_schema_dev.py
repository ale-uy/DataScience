# -*- coding: utf-8 -*-
"""
Autor: ale-uy
Fecha: 27 de Julio de 2023
Archivo: ml_schema.py
Descripción: Automatizar procesos de analisis en datos y machine learning
Licencia: Apache License Version 2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, \
    precision_score, recall_score, f1_score, roc_auc_score, r2_score
from statsmodels.stats.outliers_influence import summary_table

pd.set_option('display.max_colwidth', None) # Mostrar todo el ancho de las celdas en el DataFrame



class EDA:
    
    @classmethod
    def analizar_nulos(cls, df):
        """
        Devuelve el porcentaje de valores nulos en el total de datos para cada columna.

        Returns:
            Series: Una serie que contiene los porcentajes de valores nulos para cada columna,
                    ordenados de mayor a menor.
        """
        return df.isna().sum().sort_values(ascending=False) / df.shape[0] * 100
    
    @classmethod
    def estadisticos_numericos(cls, df):
        """
        Genera datos estadísticos de las variables numéricas en el DataFrame.

        Returns:
            DataFrame: Un DataFrame que contiene los estadísticos de las variables numéricas,
                       incluyendo count, mean, std, min, 25%, 50%, 75% y max.
        """
        df.select_dtypes('number').describe().T

    @classmethod
    def convertir_a_numericas(cls, df, target:str, metodo="ohe", drop_first=True):
        """
        Realiza la codificación de variables categóricas utilizando diferentes métodos.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables a codificar.
            target (str): El nombre de la columna objetivo que no se codificará.
            metodo (str): El método de codificación a utilizar. Opciones válidas:
                - "dummy": Codificación dummy.
                - "ohe": OneHotEncoding.
                - "label": LabelEncoder.
            drop_first (bool): Elimina la primer dummy en caso de utilizar "dummy". Por defecto es True.

        Retorna:
            pandas DataFrame: El DataFrame original con las columnas categóricas codificadas, excluyendo la columna objetivo.
        """
        if metodo == "dummy":
            """
            Realiza la codificación dummies a variables categóricas del DataFrame.

            Parámetros:
                df (pandas DataFrame): El DataFrame que contiene las variables categóricas.
                target (str): El nombre de la columna objetivo que no se codificará.
                drop_first (bool): Elimina la primer dummy. Por defecto es True

            Retorna:
                pandas DataFrame: El DataFrame original con las columnas categóricas codificadas mediante
                el método dummies, excluyendo la columna objetivo.
            """
            # Separamos la columna objetivo 'y' del resto del conjunto de datos 'X'
            y = df[target]
            X = df.drop(target, axis=1)
            # Convierto a dummy
            X = pd.get_dummies(X, drop_first=drop_first)
            # Unir el DataFrame imputado con las columnas categóricas codificadas al DataFrame 'Xy'
            Xy = pd.concat([X, y], axis=1)
            # Regresar el DataFrame 'Xy' completo con las columnas categóricas codificadas
            return Xy
        
        elif metodo == "ohe":
            """
            Este método realiza el OneHotEncoding para todas las columnas categóricas en el DataFrame, 
            excepto para la columna especificada como 'target', que se considera la variable
            objetivo y no se codificará. Las columnas categóricas se identifican automáticamente en
            función de su tipo de datos 'object'.

            Retorna:
                pandas DataFrame: El DataFrame original con las columnas categóricas codificadas mediante
                el método OneHotEncoding, excluyendo la columna objetivo.
            """
            # Crear el codificador OneHotEncoder con la opción 'drop' establecida en 'first'
            # para evitar la trampa de la codificación (colinealidad).
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            # Separamos la columna objetivo 'y' del resto del conjunto de datos 'X'
            y = df[target]
            X = df.drop(target, axis=1)
            # Obtener las columnas categóricas a codificar automáticamente en función de su tipo de datos 'object'
            object_columns = X.select_dtypes(include=['object']).columns
            # Aplicar OneHotEncoding a las columnas categóricas seleccionadas y generar un nuevo DataFrame
            X_encoded = pd.DataFrame(encoder.fit_transform(X[object_columns]))
            # Asignar nombres de columnas al nuevo DataFrame basados en los nombres de las características originales
            X_encoded.columns = encoder.get_feature_names_out(object_columns)
            # Descartar las columnas categóricas originales del DataFrame 'X'
            X = X.drop(object_columns, axis=1).reset_index(drop=True)
            # Unir el DataFrame imputado con las columnas categóricas codificadas al DataFrame 'Xy'
            Xy = pd.concat([X, X_encoded, y], axis=1)
            # Regresar el DataFrame 'Xy' completo con las columnas categóricas codificadas
            return Xy
        
        elif metodo == "label":
            """
            Realiza la codificación de variables categóricas utilizando LabelEncoder.

            Retorna:
                pandas DataFrame: El DataFrame original con las columnas categóricas codificadas
                mediante LabelEncoder, INCLUYENDO la columna objetivo.
            """
            # Creamos el DataFrame auxiliar 'X'
            X = df.copy(deep=True)

            # Obtener las columnas categóricas a codificar automáticamente en función de su tipo de datos 'object'
            object_columns = X.select_dtypes(include=['object']).columns

            # Crear un objeto LabelEncoder para cada columna categórica y transformar los datos
            label_encoders = {col: LabelEncoder() for col in object_columns}
            for col in object_columns:
                X[col] = label_encoders[col].fit_transform(X[col])

            # Regresar el DataFrame 'X' completo con las columnas categóricas codificadas
            return X

        else:
            raise ValueError("El parámetro 'metodo' debe ser uno de: 'dummy', 'ohe', 'label'.")
        
    @classmethod
    def eliminar_nulos_si(cls, df, p=0.5):
        """
        Elimina las columnas que tienen un porcentaje mayor o igual a p de valores nulos.

        Parameters:
            p (float): Porcentaje límite para considerar una columna con valores nulos.
                       Por defecto es 0.5 (50%).
        """
        nan_percentages = df.isna().mean()
        mask = nan_percentages >= p
        aux = df.loc[:, ~mask]
        return aux

    @classmethod
    def imputar_faltantes(cls, df, metodo="mm", n_neighbors=5):
        """
        ToDo
        """
        if metodo == "mm":
            """
            Imputa el valor de la mediana a los valores nulos en variables numéricas
            y la moda en variables categóricas.
            """
            aux = df.copy(deep=True)
            for col in aux.columns:
                if aux[col].dtype == 'object':
                    # Imputa la moda a los valores nulos en una columna categórica.
                    mode = aux[col].mode()[0]
                    aux[col].fillna(mode, inplace=True)
                else:
                    # Imputa la mediana a los valores nulos en una columna numérica.
                    median = aux[col].median()
                    aux[col].fillna(median, inplace=True)
            return aux

        elif metodo == "knn":
            """
            Imputa los valores faltantes en un DataFrame utilizando KNNImputer.

            Parámetros:
            df (pandas DataFrame): El DataFrame que contiene los valores faltantes.
            n_neighbors (int): El número de vecinos más cercanos a utilizar en KNNImputer.

            Retorna:
            pandas DataFrame: El DataFrame original con los valores faltantes imputados.
            """
            # Crear un objeto KNNImputer con el número de vecinos (n_neighbors) especificado
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            # Imputar los valores faltantes utilizando KNNImputer
            df_imputed = knn_imputer.fit_transform(df)
            # Reconstruir el DataFrame imputado con las mismas columnas
            df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
            return df_imputed

    @classmethod
    def eliminar_unitarios(cls, df):
        """
        Elimina las variables que tienen un solo valor.
        """
        columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
        aux = df.drop(columns_to_drop, axis=1, inplace=True)
        return aux

    @classmethod
    def shuffle_data(cls, df):
        """
        Mezcla los datos en el DataFrame de forma aleatoria.
        """
        aux = df.iloc[np.random.permutation(len(df))]
        return aux

    @classmethod
    def estandarizar_variables(cls, df, metodo="zscore"):
        """
        Estandariza las variables numéricas en un DataFrame utilizando el método especificado.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables a estandarizar.
            metodo (str): El método de estandarización a utilizar. Opciones válidas:
                - 'zscore': Estandarización utilizando el Z-Score (media 0, desviación estándar 1).
                - 'minmax': Estandarización utilizando Min-Max (rango 0 a 1).
                - 'robust': Estandarización robusta utilizando medianas y cuartiles.
                Default = zscore
        Retorna:
            pandas DataFrame: El DataFrame original con las variables numéricas estandarizadas.
        """
        aux = df.copy(deep=True)
        if metodo == 'zscore':
            scaler = StandardScaler()
        elif metodo == 'minmax':
            scaler = MinMaxScaler()
        elif metodo == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("El parámetro 'metodo' debe ser uno de: 'zscore', 'minmax', 'robust'.")
        aux[df.select_dtypes(include='number').columns] = scaler.fit_transform(df.select_dtypes(include='number'))
        return aux

    @classmethod
    def graficos_categoricos(cls, df):
        """
        Crea gráficos de barras horizontales para cada variable categórica en el DataFrame.
        """
        categorical_columns = df.select_dtypes('O').columns
        num_columns = len(categorical_columns)
        rows = (num_columns + 1) // 2
        _, ax = plt.subplots(nrows=rows, ncols=2, figsize=(16, rows * 6))
        ax = ax.flat
        for i, col in enumerate(categorical_columns):
            df[col].value_counts().plot.barh(ax=ax[i])
            ax[i].set_title(col, fontsize=12, fontweight="bold")
            ax[i].tick_params(labelsize=12)
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_histogram(cls, df, column:str):
        """
        Genera un histograma interactivo para una columna específica del DataFrame.

        Parameters:
            column (str): Nombre de la columna a visualizar en el histograma.
        """
        fig = px.histogram(df, x=column)
        fig.show()

    @classmethod
    def plot_boxplot(cls, df, x:str, y:str):
        """
        Genera un gráfico de caja interactivo para una variable y en función de otra variable x.

        Parameters:
            x (str): Nombre de la variable independiente en el gráfico de caja.
            y (str): Nombre de la variable dependiente en el gráfico de caja.
        """
        fig = px.box(df, x=x, y=y)
        fig.show()

    @classmethod
    def plot_scatter(cls, df, x:str, y:str):
        """
        Genera un gráfico de dispersión interactivo para dos variables x e y.

        Parameters:
            x (str): Nombre de la variable x en el gráfico de dispersión.
            y (str): Nombre de la variable y en el gráfico de dispersión.
        """
        fig = px.scatter(df, x=x, y=y)
        fig.show()


class ML:

    @classmethod
    def importancia_variables(cls, df, target:str, n_estimators=100, 
                              random_state=np.random.randint(1,1000), 
                              cv=5, eliminar=False, umbral=0.005):
        """
        Calcula la importancia de las variables en función de su contribución a la predicción, utiliza RandomForest.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables de entrada y la variable objetivo.
            target (str): El nombre de la columna que contiene la variable objetivo.
            n_estimators (int): Numero de arboles que se usaran para la clasificacion, 100 por defecto.
            random_state (int): semilla a usar, por defecto es un numero aleatorio.
            cv (int): Numero de subgrupos para el cross-validation, 5 por defecto.
            eliminar (bool): Si se debe eliminar las variables menos importantes. Por defecto es False.
            umbral (float): Valor umbral que determina la importancia mínima requerida para mantener una variable.
                            Por defecto es 0.005.

        Retorna:
            pandas DataFrame: Un DataFrame que contiene el ranking de importancia de cada variable.
            float: El rendimiento del modelo medido por la métrica correspondiente en el conjunto de prueba.
        """
        # Separamos la variable objetivo 'y' del resto de las variables 'X'
        X = df.drop(columns=[target])
        y = df[target]

        # Creamos y entrenamos un modelo RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf_model.fit(X, y)

        # Calculamos la precisión del modelo utilizando validación cruzada
        scoring = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy').mean()

        # Creamos un DataFrame con las variables y su importancia
        importancia_df = pd.DataFrame({'Variable': X.columns, 'Importancia': rf_model.feature_importances_})
        # Ordenamos el DataFrame por la importancia en orden descendente
        importancia_df = importancia_df.sort_values(by='Importancia', ascending=False).reset_index(drop=True)

        if eliminar:
            # Obtenemos las variables que superan el umbral de importancia
            variables_importantes = importancia_df[importancia_df['Importancia'] >= umbral]['Variable']
            # Filtramos el DataFrame original manteniendo solo las variables importantes
            df_filtrado = df[df.columns[df.columns.isin(variables_importantes) | df.columns.isin([target])]]
            return df_filtrado, scoring
        else:
            return importancia_df, scoring

    @classmethod
    def modelo_lightgbm(cls, df, target:str, tipo_problema:str, 
                        boosting_type='gbdt', test_size=0.2, num_boost_round=100, graficar=False, 
                        learning_rate=0.1, max_depth=-1, random_state=np.random.randint(1, 1000)):
        """
        Utiliza LightGBM para predecir la variable objetivo en un DataFrame.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables de entrada y la variable objetivo.
            target (str): El nombre de la columna que contiene la variable objetivo.
            tipo_problema (str): Tipo de problema: 'clasificacion' o 'regresion'.
            boosting_type (str): Posibles opciones: 'rf', 'gbdt' (default), 'goss', 'dart'.
            test_size (float): El tamaño de la muestra para el conjunto de prueba, 0.2 por defecto.
            num_boost_round (int): El número de iteraciones del algoritmo (número de árboles), 100 por defecto.
            learning_rate (float): Tasa de aprendizaje del modelo, 0.1 por defecto.
            max_depth (int): Profundidad máxima de los árboles, -1 por defecto (sin límite).
            random_state (int): Semilla a usar para la división de los datos, por defecto es un número aleatorio.

        Retorna:
            float: El rendimiento del modelo medido por la métrica correspondiente en el conjunto de prueba.
        """
        # Separamos la variable objetivo 'y' del resto de las variables 'X'
        X = df.drop(columns=[target])
        y = df[target]

        # Verificamos si la variable objetivo es numérica
        if not np.issubdtype(y.dtype, np.number):
            # Si no es numérica, la convertimos utilizando LabelEncoder
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Creamos el dataset LightGBM
        lgb_train = lgb.Dataset(X_train, y_train)

        # Parámetros del modelo LightGBM
        params = {
            'boosting_type': boosting_type,
            'learning_rate': learning_rate,
            'max_depth': max_depth
        }

        if tipo_problema == 'clasificacion':
            # Problema de clasificación binaria
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            # Entrenamos el modelo LightGBM para clasificación
            lgb_model = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
            # Realizamos predicciones en el conjunto de prueba
            y_pred = lgb_model.predict(X_test)
            y_pred_binary = np.round(y_pred)  # Convertimos las predicciones a 0 o 1
            # Calculamos las métricas de clasificación
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary)
            recall = recall_score(y_test, y_pred_binary)
            f1 = f1_score(y_test, y_pred_binary)
            roc_auc = roc_auc_score(y_test, y_pred_binary)

            # Creamos un DataFrame con las métricas
            metric = pd.DataFrame({
                'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'],
                'Valor': [accuracy, precision, recall, f1, roc_auc],
                'Explicacion': [
                    '(Exactitud) Proporción de muestras correctamente clasificadas.',
                    'Precisión Proporción de muestras positivas correctamente identificadas entre todas las muestras clasificadas como positivas.',
                    '(Sensibilidad) Proporción de muestras positivas correctamente identificadas entre todas las muestras reales positivas.',
                    'Media armónica entre precision y recall. Es útil cuando hay un desequilibrio entre las clases.',
                    'Área bajo la curva ROC, que mide la capacidad de discriminación del modelo.']
                })
            if graficar == True:
                cls._plot_clasificacion(y_test, y_pred_binary)

        elif tipo_problema == 'regresion':
            # Problema de regresión
            params['objective'] = 'regression'
            params['metric'] = 'l2'  # MSE (Error Cuadrático Medio)
            # Entrenamos el modelo LightGBM para regresión
            lgb_model = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
            # Realizamos predicciones en el conjunto de prueba
            y_pred = lgb_model.predict(X_test)
            # Calculamos el MSE del modelo en el conjunto de prueba
            metric = mean_squared_error(y_test, y_pred)
            # Calculamos el R cuadrado ajustado
            ssr = np.sum((y_test - y_pred) ** 2)
            sst = np.sum((y_test - np.mean(y_test)) ** 2)
            r_squared = 1 - (ssr / sst) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
            # Calculamos el p-valor utilizando statsmodels
            _, summary_data, _ = summary_table(lgb_model, X_test)
            p_values = summary_data[:, 4]
            # Creamos un DataFrame con las métricas y sus explicaciones
            metric = pd.DataFrame({
                'Metrica': ['MSE', 'R cuadrado ajustado', 'p-valor'],
                'Valor': [metric, r_squared, p_values],
                'Explicacion': [
                    'Error cuadrático medio entre las predicciones y los valores reales.',
                    'Proporción de la varianza total explicada por el modelo ajustado. '
                    'Tiene en cuenta la cantidad de variables independientes en el modelo.',
                    'Probabilidad de obtener el estadístico de prueba (coeficiente del modelo) '
                    'asumiendo que la verdadera relación entre la variable dependiente y las variables independientes es cero.']
            })
            if graficar == True:
                cls._plot_regresion(y_test, y_pred)

        else:
            raise ValueError("El parámetro 'tipo_problema' debe ser 'clasificacion' o 'regresion'.")

        return metric
    
    @classmethod
    def modelo_adaboost(cls, df, target:str, tipo_problema:str, 
                        test_size=0.2, n_estimators=50, learning_rate=1.0,
                        random_state=np.random.randint(1, 1000), graficar=False):
        """
        Utiliza AdaBoost para predecir la variable objetivo en un DataFrame.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables de entrada y la variable objetivo.
            target (str): El nombre de la columna que contiene la variable objetivo.
            tipo_problema (str): Tipo de problema: 'clasificacion' o 'regresion'.
            test_size (float): El tamaño de la muestra para el conjunto de prueba, 0.2 por defecto.
            n_estimators (int): El número de estimadores (número de árboles), 50 por defecto.
            learning_rate (float): Tasa de aprendizaje del modelo, 1.0 por defecto.
            random_state (int): Semilla a usar para la división de los datos, por defecto es un número aleatorio.
            graficar (bool): Si es True, se generan los gráficos correspondientes según el tipo de problema.

        Retorna:
            pd.DataFrame: Un DataFrame que contiene diferentes métricas y estadísticas del modelo.
        """
        # Separamos la variable objetivo 'y' del resto de las variables 'X'
        X = df.drop(columns=[target])
        y = df[target]

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Creamos el modelo AdaBoost
        if tipo_problema == 'clasificacion':
            model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )
        elif tipo_problema == 'regresion':
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )
        else:
            raise ValueError("El parámetro 'tipo_problema' debe ser 'clasificacion' o 'regresion'.")
        
        # Entrenamos el modelo AdaBoost
        model.fit(X_train, y_train)

        # Realizamos predicciones en el conjunto de prueba
        y_pred = model.predict(X_test)

        # Calculamos las métricas de clasificación o regresión
        if tipo_problema == 'clasificacion':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Creamos un DataFrame con las métricas y sus explicaciones
            metrics = pd.DataFrame({
                'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                'Valor': [accuracy, precision, recall, f1],
                'Explicacion': [
                    'Accuracy (Exactitud): Proporción de muestras correctamente clasificadas.',
                    'Precision (Precisión): Media ponderada de precision para cada clase.',
                    'Recall (Sensibilidad): Media ponderada de recall para cada clase.',
                    'F1-score (Puntuación F1): Media ponderada de F1-score para cada clase.']
            })
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Creamos un DataFrame con las métricas y sus explicaciones
            metrics = pd.DataFrame({
                'Metrica': ['Mean Squared Error (MSE)', 'R-cuadrado (R^2)'],
                'Valor': [mse, r2],
                'Explicacion': [
                    'Mean Squared Error (MSE): Error cuadrático medio entre las predicciones y los valores verdaderos.',
                    'R-cuadrado (R^2): Coeficiente de determinación que indica la proporción de la varianza total de la variable dependiente explicada por el modelo.']
            })

        if graficar:
            if tipo_problema == 'clasificacion':
                cls._plot_clasificacion(y_test, y_pred)
            else:
                cls._plot_regresion(y_test, y_pred)

        # Mostramos todas las explicaciones sin recortar la columna
        pd.set_option('display.max_colwidth', None)
        
        return metrics
    
    @classmethod
    def modelo_xgboost(cls, df, target:str, tipo_problema:str, 
                        test_size=0.2, num_boost_round=100,
                        learning_rate=0.1, max_depth=3, objective='binary:logistic',
                        random_state=np.random.randint(1, 1000), graficar=False):
        """
        Utiliza XGBoost para predecir la variable objetivo en un DataFrame.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables de entrada y la variable objetivo.
            target (str): El nombre de la columna que contiene la variable objetivo.
            tipo_problema (str): Tipo de problema: 'clasificacion' o 'regresion'.
            test_size (float): El tamaño de la muestra para el conjunto de prueba, 0.2 por defecto.
            num_boost_round (int): El número de iteraciones del algoritmo (número de árboles), 100 por defecto.
            learning_rate (float): Tasa de aprendizaje del modelo, 0.1 por defecto.
            max_depth (int): Profundidad máxima de los árboles, 3 por defecto.
            objective (str): Función objetivo, 'reg:squarederror' para regresión, 
                'binary:logistic' para clasificación binaria, 'multi:softmax' para clasificación multiclase.
            random_state (int): Semilla a usar para la división de los datos, por defecto es un número aleatorio.
            graficar (bool): Si es True, se generan los gráficos correspondientes según el tipo de problema.

        Retorna:
            pd.DataFrame: Un DataFrame que contiene diferentes métricas y estadísticas del modelo.
        """
        # Separamos la variable objetivo 'y' del resto de las variables 'X'
        X = df.drop(columns=[target])
        y = df[target]

        # Verificamos si la variable objetivo es numérica
        if not np.issubdtype(y.dtype, np.number):
            # Si no es numérica, la convertimos utilizando LabelEncoder
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Creamos el objeto DMatrix de XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Parámetros del modelo XGBoost
        params = {
            'objective': objective,
            'learning_rate': learning_rate,
            'max_depth': max_depth
        }

        if tipo_problema == 'clasificacion':
            # Problema de clasificación binaria o multiclase
            if objective == 'reg:squarederror':
                raise ValueError("La función objetivo debe ser 'binary:logistic' o 'multi:softmax' para un problema de clasificación.")
            
            # Entrenamos el modelo XGBoost para clasificación
            xgb_model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
            
            # Realizamos predicciones en el conjunto de prueba
            y_pred = xgb_model.predict(dtest)
            if 'binary' in objective:
                y_pred_binary = (y_pred > 0.5).astype(int)
            else:
                y_pred_binary = np.argmax(y_pred, axis=1)

            # Calculamos las métricas de clasificación
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary, average='weighted')
            recall = recall_score(y_test, y_pred_binary, average='weighted')
            f1 = f1_score(y_test, y_pred_binary, average='weighted')
            roc_auc = roc_auc_score(y_test, y_pred_binary)

            # Creamos un DataFrame con las métricas y sus explicaciones
            metrics = pd.DataFrame({
                'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC ROC'],
                'Valor': [accuracy, precision, recall, f1, roc_auc],
                'Explicacion': [
                    'Accuracy (Exactitud): Proporción de muestras correctamente clasificadas.',
                    'Precision (Precisión): Media ponderada de precision para cada clase.',
                    'Recall (Sensibilidad): Media ponderada de recall para cada clase.',
                    'F1-score (Puntuación F1): Media ponderada de F1-score para cada clase.',
                    'Área bajo la curva ROC, mide la capacidad de discriminación del modelo.']
            })
            if graficar == True:
                cls._plot_clasificacion(y_test, y_pred_binary)

        elif tipo_problema == 'regresion':
            # Problema de regresión
            objective = 'reg:squarederror'
            
            # Entrenamos el modelo XGBoost para regresión
            xgb_model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
            
            # Realizamos predicciones en el conjunto de prueba
            y_pred = xgb_model.predict(dtest)

            # Calculamos las métricas de regresión
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

            # Creamos un DataFrame con las métricas y sus explicaciones
            metrics = pd.DataFrame({
                'Metrica': ['Mean Squared Error (MSE)', 'R-cuadrado Ajustado (Adj. R^2)'],
                'Valor': [mse, adj_r2],
                'Explicacion': [
                    'Mean Squared Error (MSE): Error cuadrático medio entre las predicciones y los valores verdaderos.',
                    'R-cuadrado Ajustado (Adj. R^2): R-cuadrado ajustado que tiene en cuenta el número de variables independientes en el modelo.']
            })

        else:
            raise ValueError("El parámetro 'tipo_problema' debe ser 'clasificacion' o 'regresion'.")
        
        # Mostramos todas las explicaciones sin recortar la columna
        pd.set_option('display.max_colwidth', None)

        return metrics

    @classmethod
    def modelo_catboost(cls, df, target:str, tipo_problema:str, 
                        test_size=0.2, num_boost_round=100,
                        learning_rate=0.1, max_depth=3, graficar=False,
                        random_state=np.random.randint(1, 1000)):
        """
        Utiliza CatBoost para predecir la variable objetivo en un DataFrame.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables de entrada y la variable objetivo.
            target (str): El nombre de la columna que contiene la variable objetivo.
            tipo_problema (str): Tipo de problema: 'clasificacion' o 'regresion'.
            test_size (float): El tamaño de la muestra para el conjunto de prueba, 0.2 por defecto.
            num_boost_round (int): El número de iteraciones del algoritmo (número de árboles), 100 por defecto.
            learning_rate (float): Tasa de aprendizaje del modelo, 0.1 por defecto.
            max_depth (int): Profundidad máxima de los árboles, 3 por defecto.
            random_state (int): Semilla a usar para la división de los datos, por defecto es un número aleatorio.
            graficar (bool): Si es True, se generan los gráficos correspondientes según el tipo de problema.

        Retorna:
            pd.DataFrame: Un DataFrame que contiene diferentes métricas y estadísticas del modelo.
        """
        # Separamos la variable objetivo 'y' del resto de las variables 'X'
        X = df.drop(columns=[target])
        y = df[target]

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Creamos el modelo CatBoost
        if tipo_problema == 'clasificacion':
            loss_function = 'Logloss'
            model = cb.CatBoostClassifier(
                iterations=num_boost_round,
                learning_rate=learning_rate,
                depth=max_depth,
                loss_function=loss_function,
                random_state=random_state
            )
        elif tipo_problema == 'regresion':
            loss_function = 'RMSE'
            model = cb.CatBoostRegressor(
                iterations=num_boost_round,
                learning_rate=learning_rate,
                depth=max_depth,
                loss_function=loss_function,
                random_state=random_state
            )
        else:
            raise ValueError("El parámetro 'tipo_problema' debe ser 'clasificacion' o 'regresion'.")
        
        # Entrenamos el modelo CatBoost
        model.fit(X_train, y_train)

        # Realizamos predicciones en el conjunto de prueba
        y_pred = model.predict(X_test)

        # Calculamos las métricas de clasificación o regresión
        if tipo_problema == 'clasificacion':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_pred)

            # Creamos un DataFrame con las métricas y sus explicaciones
            metrics = pd.DataFrame({
                'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC ROC'],
                'Valor': [accuracy, precision, recall, f1, roc_auc],
                'Explicacion': [
                    'Accuracy (Exactitud): Proporción de muestras correctamente clasificadas.',
                    'Precision (Precisión): Media ponderada de precision para cada clase.',
                    'Recall (Sensibilidad): Media ponderada de recall para cada clase.',
                    'F1-score (Puntuación F1): Media ponderada de F1-score para cada clase.',
                    'Área bajo la curva ROC, mide la capacidad de discriminación del modelo.']
            })
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

            # Creamos un DataFrame con las métricas y sus explicaciones
            metrics = pd.DataFrame({
                'Metric': ['Mean Squared Error (MSE)', 'R-cuadrado Ajustado (Adj. R^2)'],
                'Valor': [mse, adj_r2],
                'Explicacion': [
                    'Mean Squared Error (MSE): Error cuadrático medio entre las predicciones y los valores verdaderos.',
                    'R-cuadrado Ajustado (Adj. R^2): R-cuadrado ajustado que tiene en cuenta el número de variables independientes en el modelo.',
                    ]
            })

        # Mostramos todas las explicaciones sin recortar la columna
        pd.set_option('display.max_colwidth', None)

        if graficar:
            if tipo_problema == 'clasificacion':
                cls._plot_clasificacion(y_test, y_pred)
            else:
                cls._plot_regresion(y_test, y_pred)

        return metrics

    @classmethod
    def _plot_clasificacion(cls, y_true, y_pred):
        """
        Genera una matriz de confusión interactiva para evaluar el rendimiento del modelo.

        Parámetros:
            y_true (array-like): Los valores verdaderos de la variable objetivo.
            y_pred (array-like): Los valores predichos por el modelo.

        Retorna:
            None
        """
        # Verificamos si los valores son numéricos o categóricos
        if not np.issubdtype(y_true.dtype, np.number):
            label_encoder = LabelEncoder()
            y_true = label_encoder.fit_transform(y_true)
            y_pred = label_encoder.transform(y_pred)

        # Calculamos la matriz de confusión
        cm = confusion_matrix(y_true, y_pred)

        # Generamos la matriz de confusión interactiva utilizando Plotly
        fig = go.Figure(data=go.Heatmap(z=cm, x=['Clase 0', 'Clase 1'], y=['Clase 0', 'Clase 1'],
                                       colorscale='YlGnBu', zmin=0, zmax=cm.max().max()))

        # Mostramos los valores dentro del gráfico
        for i in range(len(cm)):
            for j in range(len(cm)):
                fig.add_annotation(
                    x=['Clase 0', 'Clase 1'][j],
                    y=['Clase 0', 'Clase 1'][i],
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(size=14, color='white' if cm[i, j] > cm.max().max() / 2 else 'black')
                )

        fig.update_layout(title='Matriz de Confusión',
                          xaxis_title='Valores Predichos',
                          yaxis_title='Valores Verdaderos',
                          xaxis=dict(side='top'))

        fig.show()

    @classmethod
    def _plot_regresion(cls, y_true, y_pred):
        """
        Genera un gráfico de dispersión interactivo para comparar los valores verdaderos y los valores predichos en un problema de regresión.

        Parámetros:
            y_true (array-like): Los valores verdaderos de la variable objetivo.
            y_pred (array-like): Los valores predichos por el modelo.

        Retorna:
            None
        """
        # Verificamos si los valores son numéricos o categóricos
        if not np.issubdtype(y_true.dtype, np.number):
            raise ValueError("El método solo es válido para problemas de regresión con variables numéricas.")

        # Creamos un DataFrame para los valores verdaderos y predichos
        df = pd.DataFrame({'True': y_true, 'Predicted': y_pred})

        # Generamos el gráfico de dispersión interactivo utilizando Plotly
        fig = px.scatter(df, x='True', y='Predicted', labels={'True': 'Valores Verdaderos', 'Predicted': 'Valores Predichos'},
                         title='Comparación entre Valores Verdaderos y Valores Predichos (Regresión)')
        fig.add_trace(go.Scatter(x=[min(y_true), max(y_true)], y=[min(y_true), max(y_true)], mode='lines', line=dict(color='red'),
                                 name='Línea de 45°'))

        fig.show()