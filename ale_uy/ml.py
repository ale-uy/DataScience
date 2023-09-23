# -*- coding: utf-8 -*-
"""
Autor: ale-uy
Fecha: 07/2023
Actualizado: 08/2023
Version: v2
Archivo: ml_vx.py
Descripción: Automatizar procesos de machine learning
Licencia: Apache License Version 2.0
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.mixture import GaussianMixture
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, \
    precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_squared_log_error, silhouette_score
from sklearn.cluster import KMeans, DBSCAN

import joblib
import warnings

pd.set_option('display.max_colwidth', None) # Mostrar todo el ancho de las celdas en el DataFrame
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL")


class Tools:

    @classmethod
    def importancia_variables(cls, df, target:str, n_estimators=100, save_model=False, peores=0,
                              random_state=np.random.randint(1,1000), cv=5, model_filename="Random_Forest",
                              tipo_problema=None, eliminar=False, umbral=0.0001):
        """
        Calcula la importancia de las variables en función de su contribución a la predicción, utiliza RandomForest.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables de entrada y la variable objetivo.
            target (str): El nombre de la columna que contiene la variable objetivo.
            n_estimators (int): Numero de arboles que se usaran para la clasificacion, 100 por defecto.
            random_state (int): semilla a usar, por defecto es un numero aleatorio.
            cv (int): Numero de subgrupos para el cross-validation, 5 por defecto.
            peores (int): cuantas variables desde abajo mostrar (todas por default)
            save_model (bool): True para guardar el modelo (False defecto)
            eliminar (bool): Si se debe eliminar las variables menos importantes. Por defecto es False.
            tipo_problema (str): 'clasificacion' o 'regresion' (None por defecto)
            umbral (float): Valor umbral que determina la importancia mínima requerida para mantener una variable.
                            Por defecto es 0.005.

        Retorna:
            pandas DataFrame: Un DataFrame que contiene el ranking de importancia de cada variable.
            float: El rendimiento del modelo medido por la métrica correspondiente en el conjunto de prueba.
        """
        # Separamos la variable objetivo 'y' del resto de las variables 'X'
        X = df.drop(columns=[target])
        y = df[target]
        if not np.issubdtype(y.dtype, np.number):
            # Si no es numérica, la convertimos utilizando LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            y = pd.Series(y_encoded, name=target) # type: ignore

        # Creamos y entrenamos un modelo RandomForest
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state) \
            if tipo_problema=='clasificacion' else \
                RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        
        rf_model.fit(X, y)

        # Calculamos la precisión del modelo utilizando validación cruzada
        scoring = 'accuracy' if tipo_problema=='clasificacion' else 'neg_mean_squared_error'
        score = cross_val_score(rf_model, X, y, cv=cv, scoring=scoring).mean()

        # Creamos un DataFrame con las variables y su importancia
        importancia_df = pd.DataFrame({'Variable': X.columns, 'Importancia': rf_model.feature_importances_})
        # Ordenamos el DataFrame por la importancia en orden descendente
        importancia_df = importancia_df.sort_values(by='Importancia', ascending=False).reset_index(drop=True)

        if save_model and model_filename:
            # Guardar el modelo entrenado en disco
            joblib.dump(rf_model, f'{model_filename}.pkl')

        if eliminar:
            # Obtenemos las variables que superan el umbral de importancia
            variables_importantes = importancia_df[importancia_df['Importancia'] >= umbral]['Variable']
            # Filtramos el DataFrame original manteniendo solo las variables importantes
            df_filtrado = df[df.columns[df.columns.isin(variables_importantes) | df.columns.isin([target])]]
            return df_filtrado
        else:
            #return print(f'{scoring.upper()}: {score}'), importancia_df
            print(f'MSE del modelo RF: {score}')
            print()
            print(f'Las 15 peores variables en aporte: ')
            print(f'{importancia_df[-peores:]}')
        
    @classmethod
    def dividir_y_convertir_datos(cls, df, target:str, test_size=0.2, 
                      random_state=np.random.randint(1, 1000), 
                      encode_categorical=False)->tuple:
        """
        Divide los datos en conjuntos de entrenamiento y prueba y opcionalmente codifica las variables categóricas.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene los datos.
            target (str): El nombre de la columna objetivo.
            test_size (float): El tamaño del conjunto de prueba. Por defecto es 0.2.
            random_state (int): La semilla aleatoria para la división de los datos. Por defecto es un valor aleatorio.
            encode_categorical (bool): Indica si se deben codificar automáticamente las variables categóricas. Por defecto es False.
            
        Retorna:
            tuple: Una tupla que contiene los conjuntos de entrenamiento y prueba en el orden: 
            (X_train, X_test, y_train, y_test).
        """
        # Separamos la variable objetivo 'y' del resto de las variables 'X'
        X = df.drop(columns=[target])
        y = df[target]

        # Codificar automáticamente las variables categóricas utilizando pd.Categorical
        if encode_categorical:
            categorical_columns = X.select_dtypes(include=['object']).columns
            label_encoder = LabelEncoder()
            for col in categorical_columns:
                X[col] = label_encoder.fit_transform(X[col])
            # Verificamos si la variable objetivo es numérica
            if not np.issubdtype(y.dtype, np.number):
                # Si no es numérica, la convertimos utilizando LabelEncoder
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                y = pd.Series(y_encoded, name=target) # type: ignore

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test
    
    @classmethod
    def _metricas(cls, y_test, y_pred, tipo_metricas=None):
        """
        Calcula las métricas adecuadas en función del tipo de valores en y_test y y_pred.

        Para valores de clasificación:
            - Accuracy: Proporción de muestras correctamente clasificadas.
            - Precision: Proporción de muestras positivas correctamente identificadas entre todas las muestras clasificadas como positivas.
            - Recall: Proporción de muestras positivas correctamente identificadas entre todas las muestras reales positivas.
            - F1-score: Media armónica entre precisión y recall. Es útil cuando hay un desequilibrio entre las clases.
            - AUC-ROC: Área bajo la curva ROC, que mide la capacidad de discriminación del modelo.

        Para valores de regresión:
            - Mean Squared Error (MSE): Error cuadrático medio entre las predicciones y los valores verdaderos.
            - R-cuadrado (R^2): Coeficiente de determinación que indica la proporción de la varianza total de la variable dependiente explicada por el modelo.

        Parámetros:
            y_test (array-like): Valores verdaderos de la variable objetivo (ground truth).
            y_pred (array-like): Valores predichos por el modelo.
            tipo_metricas (str): Podemos elegir manualmente si es 'clasificacion' o 'regresion'.

        Retorna:
            pandas DataFrame: Un DataFrame que contiene las métricas y sus respectivos valores, junto con una breve explicación para cada métrica.
        """

        if tipo_metricas == "clasificacion":
            # Valores de clasificación
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            metric_df = pd.DataFrame({
                'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'],
                'Valor': [accuracy, precision, recall, f1, roc_auc],
                'Explicacion': [
                    '(Exactitud) Proporción de muestras correctamente clasificadas.',
                    'Proporción de muestras positivas correctamente identificadas entre todas las muestras clasificadas como positivas.',
                    '(Sensibilidad) Proporción de muestras positivas correctamente identificadas entre todas las muestras reales positivas.',
                    'Media armónica entre precisión y recall. Es útil cuando hay un desequilibrio entre las clases.',
                    'Área bajo la curva ROC, que mide la capacidad de discriminación del modelo.']
            })

        elif tipo_metricas == "regresion":
            # Valores de regresión
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            try:
                rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
            except Exception:
                rmsle = 'N/a'
            r2_adj = r2_score(y_test, y_pred, multioutput='variance_weighted')
            rmse = np.sqrt(mse)

            metric_df = pd.DataFrame({
                'Metrica': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 
                            'Mean Absolute Error (MAE)', 'Root Mean Squared Logarithmic Error (RMSLE)', 
                            'Adjusted R-cuadrado (R^2 ajustado)'],
                'Valor': [mse, rmse, mae, rmsle, r2_adj],
                'Explicacion': [
                    'Mean Squared Error (MSE): Error cuadrático medio entre las predicciones y los valores verdaderos.',
                    'Root Mean Squared Error (RMSE): Raíz cuadrada del MSE, indica el error promedio de las predicciones.',
                    'Mean Absolute Error (MAE): Error absoluto medio entre las predicciones y los valores verdaderos.',
                    'Root Mean Squared Logarithmic Error (RMSLE): Raíz cuadrada del error logarítmico cuadrático medio entre las predicciones y los valores verdaderos.',
                    'Adjusted R-cuadrado (R^2 ajustado): R-cuadrado ajustado que penaliza la adición de variables irrelevantes en el modelo.']
            })

        else:
            raise ValueError("Los valores para 'tipo_metrica' deben ser 'clasificacion' o 'regresion'.")

        return metric_df

    @classmethod
    def _busqueda_rejilla(cls, model, param_grid, X_train, y_train, scoring='accuracy', cv=5):
        """
        Realiza una búsqueda de hiperparámetros utilizando GridSearchCV.

        Parámetros:
            model: El estimador del modelo que deseas ajustar.
            param_grid: Un diccionario con los hiperparámetros y sus posibles valores.
            X_train: Conjunto de entrenamiento de características.
            y_train: Etiquetas del conjunto de entrenamiento.
            scoring: La métrica de evaluación. Por defecto es 'accuracy'.
            cv: Número de particiones para validación cruzada. Por defecto es 5.

        Retorna:
            dict: Un diccionario que contiene los mejores hiperparámetros encontrados y el mejor puntaje.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        result = {
            'best_params': best_params,
            'best_score': best_score
        }

        return result
    
    @classmethod
    def _busqueda_aleatoria(cls, model, param_dist, X_train, y_train, scoring='accuracy', cv=5, n_iter=10):
        """
        Realiza una búsqueda de hiperparámetros utilizando RandomizedSearchCV.

        Parámetros:
            model: El estimador del modelo que deseas ajustar.
            param_dist: Un diccionario con los hiperparámetros y sus distribuciones para muestreo.
            X_train: Conjunto de entrenamiento de características.
            y_train: Etiquetas del conjunto de entrenamiento.
            scoring: La métrica de evaluación. Por defecto es 'accuracy'.
            cv: Número de particiones para validación cruzada. Por defecto es 5.
            n_iter: Número de combinaciones de hiperparámetros a probar. Por defecto es 10.

        Retorna:
            dict: Un diccionario que contiene los mejores hiperparámetros encontrados y el mejor puntaje.
        """
        random_search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_dist, 
            scoring=scoring, cv=cv, 
            n_iter=n_iter)
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_score = random_search.best_score_

        result = {
            'best_params': best_params,
            'best_score': best_score
        }

        return result

    @classmethod
    def generar_clusters(cls, df, k=5, eps=0.5, min_samples=5, dbscan=False, grafico=False, random_state=np.random.randint(1,1000)):
        """
        Aplica el algoritmo K-Means o DBSCAN a un DataFrame y devuelve una serie \
            con el número de cluster al que pertenece cada observación.
        
        :param df: DataFrame con los datos para el análisis.
        :param k: Número de clusters a generar. 5 por defecto (recomendable usar Graphs.plot_cluster(df))
        :param eps: Determina el radio de búsqueda alrededor de cada punto en el espacio de características.
        :param min_samples: Especifica el número mínimo de puntos dentro del radio eps \
            para que un punto sea considerado núcleo.
        :param algorithm: Algoritmo de clustering a utilizar ('kmeans' o 'dbscan').
        :param grafico: Si True, genera un gráfico de los clusters.
        :param random_state (opt): Semilla a utilizar, aleatoria por defec
        to.

        :return: Serie con los números de cluster.
        """
        
        if dbscan:
            # Aplicar DBSCAN clustering
            name = 'DBSCAN'
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(df)

        else:
            # Aplicar K-Means clustering
            name = 'K-MEANS'
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(df)
        
        # Generar el gráfico si se solicita
        if grafico:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='viridis')
            plt.title(f'Clustering con {name}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()

        # Crear una serie con los números de cluster
        cluster_series = pd.Series(labels, name='Clusters', index=df.index)

        return cluster_series

    @classmethod
    def generar_soft_clusters(cls, df, num_clusters=5, grafico=False, random_state=np.random.randint(1, 1000), n_init=10):
        """
        Aplica Gaussian Mixture Models (GMM) a un DataFrame y devuelve las probabilidades de pertenencia \
            a cada uno de los clusters.

        :param df: DataFrame con los datos para el análisis.
        :param num_clusters: Número de clusters a generar.
        :param grafico: Si True, genera un gráfico de los clusters.
        :param random_state (opt): Semilla a utilizar, aleatoria por defecto.
        :param n_init (opt): número de veces que el algoritmo EM (Expectation-Maximization) \
            intentará converger a un resultado óptimo, 10 por defecto.

        :return: DataFrame con las probabilidades de pertenencia a cada cluster.
        
        Nota: Los datos deben estar escalados para obtener buen rendimiento
        """


        # Aplicar Gaussian Mixture Models
        gmm = GaussianMixture(n_components=num_clusters, random_state=random_state, n_init=n_init)
        gmm.fit(df)
        probabilities = gmm.predict_proba(df)

        # Generar el gráfico si se solicita
        if grafico:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=probabilities, cmap='viridis')
            plt.title('Soft Clustering con Gaussian Mixture Models')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.colorbar()
            plt.show()

        # Crear un DataFrame con las probabilidades de pertenencia a cada cluster
        cluster_probabilities = pd.DataFrame(probabilities, columns=[f'Cluster_{i}' for i in range(num_clusters)], index=df.index)

        return cluster_probabilities


class Graphs_ml:

    @classmethod
    def plot_clasificacion(cls, y_true, y_pred)->None:
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
            y_true = pd.Categorical(y_true)
        if not np.issubdtype(y_pred.dtype, np.number):
            y_pred = pd.Categorical(y_pred)

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
    def plot_regresion(cls, y_true, y_pred)->None:
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

    @classmethod
    def plot_cluster(cls, df, random_state=np.random.randint(1,1000)):
        """
        Realiza el análisis de clustering utilizando el Método del Codo y el Puntaje de Silueta.
        
        :param df: DataFrame que contiene los datos para el análisis.
        :param random_state (opcional): valor de la semilla a utilizar.

        Ejemplo de uso: 
            Graphs.plot_cluster(df). El df debe estar limpio (sin faltantes ni variuables categoricas).

        Nota: Usar datos escalados para mejor resultado
        """

        # Método del codo (Elbow Method)
        def elbow_method(data, max_clusters):
            distortions = []
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
                kmeans.fit(data)
                distortions.append(kmeans.inertia_)
            return distortions

        max_clusters = 10
        distortions = elbow_method(df, max_clusters)

        # Encontrar el número óptimo de clusters usando silhouette score
        silhouette_scores = []
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            labels = kmeans.fit_predict(df)
            silhouette_avg = silhouette_score(df, labels)
            silhouette_scores.append(silhouette_avg)

        # Crear una figura con dos subplots en forma vertical
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

        # Graficar el codo en el primer subplot
        ax1.plot(range(1, max_clusters + 1), distortions, marker='o')
        ax1.set_title('Método del Codo (Elbow Method)')
        ax1.set_xlabel('Numbero de Clusters')
        ax1.set_ylabel('Distorción')

        # Graficar silhouette scores en el segundo subplot
        ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        ax2.set_title('Puntaje de Silueta (Silhouette Score")')
        ax2.set_xlabel('Número de Clusters')
        ax2.set_ylabel('Puntaje de Silueta')

        # Ajustar los espacios entre los subplots
        plt.tight_layout()
        plt.show()


class ML(Tools):

    @classmethod
    def modelo_lightgbm(cls, df, target:str, tipo_problema:str, random_state=np.random.randint(1,1000),
                        num_leaves=20, num_boost_round=100, graficar=False, test_size=0.2, cv=5,
                        learning_rate=0.1, max_depth=-1, save_model=False, model_filename='lightgbm', 
                        encode_categorical=False, grid=False, boosting_type='gbdt', n_iter=10):
        """
        Utiliza LightGBM para predecir la variable objetivo en un DataFrame.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables de entrada y la variable objetivo.
            target (str): El nombre de la columna que contiene la variable objetivo.
            tipo_problema (str): Tipo de problema: 'clasificacion' o 'regresion'.
            num_leaves (int): Número máximo de hojas en cada árbol. Controla la complejidad del modelo. Defecto 20.
            num_boost_round (int): El número de iteraciones del algoritmo (número de árboles), 100 por defecto.
            learning_rate (float): Tasa de aprendizaje del modelo, 0.1 por defecto.
            max_depth (int): Profundidad máxima de los árboles, -1 por defecto (sin límite).
            save_model (bool): Si es True, el modelo entrenado se guardará en disco. Por defecto es False.
            model_filename (str): El nombre del archivo para guardar el modelo. Requerido si save_model es True.
            grid (bool): Indica si se debe realizar una búsqueda de hiperparámetros utilizando GridSearch.
                Si es True, se realizará una búsqueda exhaustiva de hiperparámetros para optimizar el rendimiento
                del modelo LightGBM. Si es False (por defecto), no se realizará la búsqueda de hiperparámetros.
            boosting_type (str): Tipo de algoritmo de boosting a utilizar.
                Opciones disponibles:
                - 'gbdt': Gradient Boosting Decision Tree (por defecto).
                - 'dart': Dropouts meet Multiple Additive Regression Trees.
                - 'goss': Gradient-based One-Side Sampling.
            NOTA: para cargar modelo hacer:
                import joblib

                # Ruta y nombre del archivo donde se guardó el modelo
                model_filename = "nombre_del_archivo.pkl"

                # Cargar el modelo
                loaded_model = joblib.load(model_filename)

                # Ahora puedes utilizar el modelo cargado para hacer predicciones
                # Supongamos que tienes un conjunto de datos 'X_test' para hacer predicciones
                y_pred = loaded_model.predict(X_test)

        Retorna:
            print(pd.DataFrame): Un DataFrame que contiene diferentes métricas y estadísticas del modelo.
            modelo LightGBM: El Modelo entrenado.
        """
        
        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = cls.dividir_y_convertir_datos(df, target,
                                                                           test_size=test_size,
                                                                           random_state=random_state,
                                                                           encode_categorical=encode_categorical)

        if grid:
            # Definir el espacio de búsqueda de hiperparámetros
            params = {
                'boosting_type': ['gbdt', 'dart', 'goss'],
                'num_boost_round': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [2, 3, 5],
            }
            # Definir el estimador y el scoring según el tipo de problema
            estimator = lgb.LGBMClassifier() if tipo_problema=='clasificacion' else lgb.LGBMRegressor()
            scoring = 'neg_log_loss' if tipo_problema=='clasificacion' else 'neg_mean_squared_error'
            # Realizar búsqueda de rejilla utilizando el método de Tools
            best_params = cls._busqueda_aleatoria(estimator, 
                                            params, 
                                            X_train, y_train,
                                            scoring, 
                                            cv=cv,
                                            n_iter=n_iter)
            # Mostramos los valores seleccionados por GridSearch
            print(pd.DataFrame(best_params))
            # Utilizar los mejores hiperparámetros encontrados
            params = best_params['best_params']

        else:
            # Parámetros manuales del modelo LightGBM
            params = {
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'boosting_type': boosting_type,
                'num_boost_round': num_boost_round
            }

        if tipo_problema == 'clasificacion':
            # Problema de clasificación binaria
            params['objective'] = ['binary'] if y_train.nunique() == 2 else ['multiclass']
            params['metric'] = ['binary_logloss'] if y_train.nunique() == 2 else ['multi_logloss']
            # Crear el modelo LightGBM con los mejores hiperparámetros y entrenarlo
            lgb_model = lgb.LGBMClassifier(**params)
            tipo_metricas='clas'

        elif tipo_problema == 'regresion':
            # Problema de regresión
            params['objective'] = 'regression'
            params['metric'] = 'l2'  # MSE (Error Cuadrático Medio)
            # Crear el modelo LightGBM con los mejores hiperparámetros y entrenarlo
            lgb_model = lgb.LGBMRegressor(**params)
            tipo_metricas='reg'

        else:
            raise ValueError("El parámetro 'tipo_problema' debe ser 'clasificacion' o 'regresion'.")

        # Entrenar el modelo
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=50) # type: ignore
        
        # Realizamos predicciones en el conjunto de prueba
        y_pred = lgb_model.predict(X_test)
        
        # Calculamos las métricas de clasificación
        metrics = cls._metricas(y_test, y_pred, tipo_metricas=tipo_metricas)
        
        if graficar and tipo_problema == 'clasificacion':
            Graphs.plot_clasificacion(y_test, y_pred)
        elif graficar and tipo_problema == 'regresion':
            Graphs.plot_regresion(y_test, y_pred)

        if save_model and model_filename:
            # Guardar el modelo entrenado en disco
            joblib.dump(lgb_model, f'{model_filename}.pkl')

        print(metrics)
        return lgb_model

    @classmethod
    def modelo_xgboost(cls, df, target:str, tipo_problema:str, test_size=0.2, cv=5,
                n_estimators=100, save_model=False, model_filename='xgboost',
                learning_rate=0.1, max_depth=3, random_state=np.random.randint(1, 1000),
                graficar=False, grid=False, n_iter=10):
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
            NOTA: para cargar modelo hacer:
                import joblib

                # Ruta y nombre del archivo donde se guardó el modelo
                model_filename = "nombre_del_archivo.pkl"

                # Cargar el modelo
                loaded_model = joblib.load(model_filename)

                # Ahora puedes utilizar el modelo cargado para hacer predicciones
                # Supongamos que tienes un conjunto de datos 'X_test' para hacer predicciones
                y_pred = loaded_model.predict(X_test)

        Retorna:
            print(pd.DataFrame): Un DataFrame que contiene diferentes métricas y estadísticas del modelo.
            modelo XGBoost: El Modelo entrenado.
        """

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = cls.dividir_y_convertir_datos(df,target,test_size=test_size,random_state=random_state)

        if grid:
            # Parámetros del modelo XGBoost
            params = {
                'booster': ['gbtree', 'gblinear', 'dart'],
                'n_estimators': [50, 100, 200],
                'max_depth': [1, 2, 3, 5],  # Profundidad máxima de los árboles
                'learning_rate': [0.1, 0.06, 0.03, 0.01],  # Tasa de aprendizaje
                'subsample': [0.1, 0.2, 0.5, 1.0]  # Tamaño del submuestreo
            }
            # Definir el estimador y el scoring según el tipo de problema
            estimator = xgb.XGBClassifier() if tipo_problema=='clasificacion' else xgb.XGBRegressor()
            scoring = 'neg_log_loss' if tipo_problema=='clasificacion' else 'neg_mean_squared_error'
            # Realizar búsqueda de rejilla utilizando el método de Tools
            best_params = cls._busqueda_aleatoria(estimator, 
                                            params, 
                                            X_train, y_train,
                                            scoring, 
                                            cv=cv,
                                            n_iter=n_iter)
            # Mostramos los valores seleccionados por GridSearch
            print(pd.DataFrame(best_params))
            # Utilizar los mejores hiperparámetros encontrados
            params = best_params['best_params']

        else:
            params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators
            }

        if tipo_problema == 'clasificacion':
            # Problema de clasificación binaria o multiclase
            params['objective'] = 'binary:logistic' if y_test.nunique() == 2 else 'multi:softmax'
            
            # Entrenamos el modelo XGBoost para clasificación
            xgb_model = xgb.XGBClassifier(**params)
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
            
            # Realizamos predicciones en el conjunto de prueba
            y_pred = xgb_model.predict(X_test)
            if params['objective'] == 'binary:logistic':
                y_pred_binary = np.where(y_pred > 0.5, 1, 0)
            else:
                y_pred_binary = np.argmax(y_pred, axis=1)
            metrics = cls._metricas(y_test,y_pred_binary,tipo_metricas='clas')
            if graficar == True:
                Graphs.plot_clasificacion(y_test, y_pred_binary)

        elif tipo_problema == 'regresion':
            # Problema de regresión
            params['objective'] = 'reg:squarederror'

            # Entrenamos el modelo XGBoost para regresión
            xgb_model = xgb.XGBRegressor(**params)
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
            
            # Realizamos predicciones en el conjunto de prueba
            y_pred = xgb_model.predict(X_test)
            metrics = cls._metricas(y_test, y_pred,tipo_metricas='reg')

        else:
            raise ValueError("El parámetro 'tipo_problema' debe ser 'clasificacion' o 'regresion'.")
        
        if save_model and model_filename:
            # Guardar el modelo entrenado en disco
            joblib.dump(xgb_model, f'{model_filename}.pkl')
        
        print(metrics)
        return xgb_model
    
    @classmethod
    def modelo_catboost(cls, df, target:str, tipo_problema:str, test_size=0.2, n_iter=10,
                        num_boost_round=100, learning_rate=0.1, max_depth=3, cv=3,
                        random_state=np.random.randint(1, 1000), graficar=False,
                        save_model=False, model_filename='catboost', grid=False):
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
            grid (bool): Si es True, se usa  gridsearch para buscar mejores valores en los hiperparametros. Default es False.
            
            NOTA: para cargar modelo hacer:
                import joblib

                # Ruta y nombre del archivo donde se guardó el modelo
                model_filename = "nombre_del_archivo.pkl"

                # Cargar el modelo
                loaded_model = joblib.load(model_filename)

                # Ahora puedes utilizar el modelo cargado para hacer predicciones
                # Supongamos que tienes un conjunto de datos 'X_test' para hacer predicciones
                y_pred = loaded_model.predict(X_test)

        Retorna:
            print(pd.DataFrame): Un DataFrame que contiene diferentes métricas y estadísticas del modelo.
            modelo CatBoost: El Modelo entrenado.
        """

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = cls.dividir_y_convertir_datos(df,target,test_size=test_size,random_state=random_state)
        if grid:
            params = {
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'depth': [2, 3, 5],
                'num_boost_round': [50, 100, 200]
            }
            # Definir el estimador y el scoring según el tipo de problema
            estimator = cb.CatBoostClassifier() if tipo_problema=='clasificacion' else cb.CatBoostRegressor()
            scoring = 'neg_log_loss' if tipo_problema=='clasificacion' else 'neg_mean_squared_error'
            # Realizar búsqueda de rejilla utilizando el método de Tools
            best_params = cls._busqueda_aleatoria(estimator, 
                                            params, 
                                            X_train, y_train,
                                            scoring, 
                                            cv=cv,
                                            n_iter=n_iter)
            # Mostramos los valores seleccionados por GridSearch
            print(pd.DataFrame(best_params))
            # Utilizar los mejores hiperparámetros encontrados
            params = best_params['best_params']
        else:
            params = {
                'num_boost_round': num_boost_round,
                'learning_rate': learning_rate,
                'depth': max_depth,
                'random_state': random_state
            }

        # Creamos el modelo CatBoost
        if tipo_problema == 'clasificacion':
            params['loss_function'] = 'Logloss'
            model = cb.CatBoostClassifier(**params)
        elif tipo_problema == 'regresion':
            params['loss_function'] = 'RMSE'
            model = cb.CatBoostRegressor(**params)
        else:
            raise ValueError("El parámetro 'tipo_problema' debe ser 'clasificacion' o 'regresion'.")

        # Entrenamos el modelo CatBoost
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10, verbose=50)
        # Realizamos predicciones en el conjunto de prueba
        y_pred = model.predict(X_test)
        # Calculamos las métricas de clasificación o regresión
        if tipo_problema == 'clasificacion':
            metrics = cls._metricas(y_test,y_pred,tipo_metricas='clas')
        else:
            metrics = cls._metricas(y_test,y_pred,tipo_metricas='reg')

        if graficar:
            if tipo_problema == 'clasificacion':
                Graphs.plot_clasificacion(y_test, y_pred)
            else:
                Graphs.plot_regresion(y_test, y_pred)

        if save_model and model_filename:
            # Guardar el modelo entrenado en disco
            joblib.dump(model, f'{model_filename}.pkl')

        print(metrics)
        return model
    