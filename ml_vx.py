# -*- coding: utf-8 -*-
"""
Autor: ale-uy
Fecha: 31 Julio 2023
Actualizado: 31 Julio 2023
Archivo: ml_vx.py
Descripción: Automatizar procesos de machine learning
Licencia: Apache License Version 2.0
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, \
    mean_squared_log_error, precision_score, recall_score, f1_score, roc_auc_score, r2_score
import joblib

pd.set_option('display.max_colwidth', None) # Mostrar todo el ancho de las celdas en el DataFrame


class Tools:

    @classmethod
    def importancia_variables(cls, df, target:str, n_estimators=100, 
                              random_state=np.random.randint(1,1000), 
                              cv=5, eliminar=False, umbral=0.005)->tuple:
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
    def dividir_y_convertir_datos(cls, df, target:str, test_size=0.2, 
                      random_state=np.random.randint(1, 1000), 
                      encode_categorical=True)->tuple:
        """
        Divide los datos en conjuntos de entrenamiento y prueba y opcionalmente codifica las variables categóricas.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene los datos.
            target (str): El nombre de la columna objetivo.
            test_size (float): El tamaño del conjunto de prueba. Por defecto es 0.2.
            random_state (int): La semilla aleatoria para la división de los datos. Por defecto es un valor aleatorio.
            encode_categorical (bool): Indica si se deben codificar automáticamente las variables categóricas. Por defecto es True.
            
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
            X[categorical_columns] = X[categorical_columns].apply(lambda col: pd.Categorical(col))
            # Verificamos si la variable objetivo es numérica
            if not np.issubdtype(y.dtype, np.number):
                # Si no es numérica, la convertimos utilizando LabelEncoder
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test
    
    @classmethod
    def metricas(cls, y_test, y_pred, tipo_metricas=None):
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
            tipo_metricas (str): Podemos elegir manualmente si es 'clas' o 'reg' (clasificacion o regresion).

        Retorna:
            pandas DataFrame: Un DataFrame que contiene las métricas y sus respectivos valores, junto con una breve explicación para cada métrica.
        """

        # Verificar el tipo de valores en y_test y y_pred
        y_type = type(y_test[0])
        y_pred_type = type(y_pred[0])

        if (y_type == int and y_pred_type == int) or tipo_metricas == "clas":
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
                    'Precisión Proporción de muestras positivas correctamente identificadas entre todas las muestras clasificadas como positivas.',
                    '(Sensibilidad) Proporción de muestras positivas correctamente identificadas entre todas las muestras reales positivas.',
                    'Media armónica entre precisión y recall. Es útil cuando hay un desequilibrio entre las clases.',
                    'Área bajo la curva ROC, que mide la capacidad de discriminación del modelo.']
            })

        elif (y_type == float and y_pred_type == float) or tipo_metricas == "reg":
            # Valores de regresión
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
            r2_adj = r2_score(y_test, y_pred, multioutput='variance_weighted')
            rmse = np.sqrt(mse)

            metric_df = pd.DataFrame({
                'Metrica': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 
                            'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)', 
                            'Root Mean Squared Logarithmic Error (RMSLE)', 'Adjusted R-cuadrado (R^2 ajustado)'],
                'Valor': [mse, rmse, mae, mape, rmsle, r2_adj],
                'Explicacion': [
                    'Mean Squared Error (MSE): Error cuadrático medio entre las predicciones y los valores verdaderos.',
                    'Root Mean Squared Error (RMSE): Raíz cuadrada del MSE, indica el error promedio de las predicciones.',
                    'Mean Absolute Error (MAE): Error absoluto medio entre las predicciones y los valores verdaderos.',
                    'Mean Absolute Percentage Error (MAPE): Porcentaje promedio del error absoluto entre las predicciones y los valores verdaderos.',
                    'Root Mean Squared Logarithmic Error (RMSLE): Raíz cuadrada del error logarítmico cuadrático medio entre las predicciones y los valores verdaderos.',
                    'Adjusted R-cuadrado (R^2 ajustado): R-cuadrado ajustado que penaliza la adición de variables irrelevantes en el modelo.']
            })

        else:
            raise ValueError("Los valores de y_test y y_pred deben ser del mismo tipo (int para clasificación o float para regresión).")

        return metric_df


class Graphs:

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


class ML:

    @classmethod
    def modelo_lightgbm(cls, df, target:str, tipo_problema:str, random_state=np.random.randint(1,1000),
                        boosting_type='gbdt', num_boost_round=100, graficar=False, test_size=0.2, 
                        learning_rate=0.1, max_depth=-1, save_model=False, model_filename='lightgbm'):
        """
        Utiliza LightGBM para predecir la variable objetivo en un DataFrame.

        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables de entrada y la variable objetivo.
            target (str): El nombre de la columna que contiene la variable objetivo.
            tipo_problema (str): Tipo de problema: 'clasificacion' o 'regresion'.
            boosting_type (str): Posibles opciones: 'rf', 'gbdt' (default), 'goss', 'dart'.
            num_boost_round (int): El número de iteraciones del algoritmo (número de árboles), 100 por defecto.
            learning_rate (float): Tasa de aprendizaje del modelo, 0.1 por defecto.
            max_depth (int): Profundidad máxima de los árboles, -1 por defecto (sin límite).
            save_model (bool): Si es True, el modelo entrenado se guardará en disco. Por defecto es False.
            model_filename (str): El nombre del archivo para guardar el modelo. Requerido si save_model es True.
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
            tuple: Una tupla que contiene las métricas y las predicciones en el orden: (metrics, y_pred).
        """

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = Tools.dividir_y_convertir_datos(df, target,test_size=test_size,random_state=random_state)

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
            metrics = Tools.metricas(y_test, y_pred_binary, tipo_metricas='clas')
            if graficar:
                Graphs.plot_clasificacion(y_test, y_pred_binary)

        elif tipo_problema == 'regresion':
            # Problema de regresión
            params['objective'] = 'regression'
            params['metric'] = 'l2'  # MSE (Error Cuadrático Medio)
            # Entrenamos el modelo LightGBM para regresión
            lgb_model = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
            # Realizamos predicciones en el conjunto de prueba
            y_pred = lgb_model.predict(X_test)
            # Calculamos las metricas
            metrics = Tools.metricas(y_test, y_pred, tipo_metricas='reg')
            if graficar:
                Graphs.plot_regresion(y_test, y_pred)

        else:
            raise ValueError("El parámetro 'tipo_problema' debe ser 'clasificacion' o 'regresion'.")

        if save_model and model_filename:
            # Guardar el modelo entrenado en disco
            joblib.dump(lgb_model, f'{model_filename}.pkl')

        return metrics

    @classmethod
    def modelo_xgboost(cls, df, target:str, tipo_problema:str, test_size=0.2,
                num_boost_round=100, save_model=False, model_filename='xgboost',
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
            pd.DataFrame: Un DataFrame que contiene diferentes métricas y estadísticas del modelo.
        """

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = Tools.dividir_y_convertir_datos(df,target,test_size=test_size,random_state=random_state)
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
            metrics = Tools.metricas(y_test,y_pred_binary,tipo_metricas='clas')
            if graficar == True:
                Graphs.plot_clasificacion(y_test, y_pred_binary)
        elif tipo_problema == 'regresion':
            # Problema de regresión
            objective = 'reg:squarederror'
            
            # Entrenamos el modelo XGBoost para regresión
            xgb_model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
            
            # Realizamos predicciones en el conjunto de prueba
            y_pred = xgb_model.predict(dtest)
            metrics = Tools.metricas(y_test, y_pred,tipo_metricas='reg')
        else:
            raise ValueError("El parámetro 'tipo_problema' debe ser 'clasificacion' o 'regresion'.")
        
        if save_model and model_filename:
            # Guardar el modelo entrenado en disco
            joblib.dump(xgb_model, f'{model_filename}.pkl')
        
        return metrics
    
    @classmethod
    def modelo_catboost(cls, df, target:str, tipo_problema:str, test_size=0.2,
                        num_boost_round=100, learning_rate=0.1, max_depth=3,
                        random_state=np.random.randint(1, 1000), graficar=False,
                        save_model=False, model_filename='catoost'):
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
            pd.DataFrame: Un DataFrame que contiene diferentes métricas y estadísticas del modelo.
        """

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = Tools.dividir_y_convertir_datos(df,target,test_size=test_size,random_state=random_state)
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
            metrics = Tools.metricas(y_test,y_pred,tipo_metricas='clas')
        else:
            metrics = Tools.metricas(y_test,y_pred,tipo_metricas='reg')

        if graficar:
            if tipo_problema == 'clasificacion':
                Graphs.plot_clasificacion(y_test, y_pred)
            else:
                Graphs.plot_regresion(y_test, y_pred)

        if save_model and model_filename:
            # Guardar el modelo entrenado en disco
            joblib.dump(model, f'{model_filename}.pkl')

        return metrics
    
    @classmethod
    def modelo_adaboost(cls, df, target:str, tipo_problema:str, 
                        test_size=0.2, n_estimators=50, learning_rate=1.0,
                        random_state=np.random.randint(1, 1000), graficar=False,
                        save_model=False, model_filename='catoost'):
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
            pd.DataFrame: Un DataFrame que contiene diferentes métricas y estadísticas del modelo.
        """

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = Tools.dividir_y_convertir_datos(df,target,test_size=test_size, random_state=random_state)
        
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
            metrics = Tools.metricas(y_test,y_pred,tipo_metricas='clas')
        else:
            metrics = Tools.metricas(y_test,y_pred,tipo_metricas='reg')

        if graficar:
            if tipo_problema == 'clasificacion':
                Graphs.plot_clasificacion(y_test, y_pred)
            else:
                Graphs.plot_regresion(y_test, y_pred)

        if save_model and model_filename:
            # Guardar el modelo entrenado en disco
            joblib.dump(model, f'{model_filename}.pkl')

        return metrics