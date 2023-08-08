# -*- coding: utf-8 -*-
"""
Autor: ale-uy
Fecha: 2 Mayo 2023
Actualizado: 7 Agosto 2023
Version: v2
Archivo: eda_vx.py
Descripción: Automatizar procesos de analisis y limpieza dn datos
Licencia: Apache License Version 2.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.utils import resample
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler


class Eda:

    @classmethod
    def analizar_nulos(cls, df)->None:
        """
        Devuelve el porcentaje de valores nulos en el total de datos para cada columna.
        Devuelve:
            Series: Una serie que contiene los porcentajes de valores nulos para cada columna,
                    ordenados de mayor a menor.
        """
        print(df.isna().sum().sort_values(ascending=False) / df.shape[0] * 100)

    @classmethod
    def estadisticos_numericos(cls, df)->None:
        """
        Genera datos estadísticos de las variables numéricas en el DataFrame.
        Returns:
            DataFrame: Un DataFrame que contiene los estadísticos de las variables numéricas,
                       incluyendo count, mean, std, min, 25%, 50%, 75% y max.
        """
        print(df.select_dtypes('number').describe().T)

    @classmethod
    def convertir_a_numericas(cls, df, target:str=None, metodo="ohe", drop_first=True):
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

        # Separamos la columna objetivo 'y' del resto del conjunto de datos 'X'
        y = df[target]
        if not np.issubdtype(y.dtype, np.number):
            # Si y no es numérica, la convertimos utilizando LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(df[target])
            y = pd.Series(y_encoded, name=target)
        X = df.drop(target, axis=1)

        # Utilizamos los metodos
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
            # Convierto a dummy
            X = pd.get_dummies(X, drop_first=drop_first)
            # Unir el DataFrame imputado con las columnas categóricas codificadas al DataFrame 'Xy'
            df_codificado = pd.concat([X, y], axis=1)
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
            # Crear el codificador OneHotEncoder con la opción 'drop' establecida en 'first' \
            # para evitar la trampa de la codificación (colinealidad).
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            # Obtener las columnas categóricas a codificar automáticamente en función de su tipo de datos 'object'
            object_columns = X.select_dtypes(include=['object']).columns
            # Aplicar OneHotEncoding a las columnas categóricas seleccionadas y generar un nuevo DataFrame
            X_encoded = pd.DataFrame(encoder.fit_transform(X[object_columns]))
            # Asignar nombres de columnas al nuevo DataFrame basados en los nombres de las características originales
            X_encoded.columns = encoder.get_feature_names_out(object_columns)
            # Descartar las columnas categóricas originales del DataFrame 'X'
            X = X.drop(object_columns, axis=1).reset_index(drop=True)
            # Restablecer el índice de X y X_encode para que comience desde 1 en lugar de 0
            X.index = y.index # SOLUCION PROVISORIA #
            X_encoded.index = y.index # SOLUCION PROVISORIA #
            # Unir el DataFrame imputado con las columnas categóricas codificadas al DataFrame 'Xy'
            df_codificado = pd.concat([X, X_encoded, y], axis=1)
        elif metodo == "label":
            """
            Realiza la codificación de variables categóricas utilizando LabelEncoder.
            Retorna:
                pandas DataFrame: El DataFrame original con las columnas categóricas codificadas
                mediante LabelEncoder, INCLUYENDO la columna objetivo.
            """
            # Creamos el DataFrame auxiliar 'X'
            df_codificado = df.copy(deep=True)
            # Obtener las columnas categóricas a codificar automáticamente en función de su tipo de datos 'object'
            object_columns = df_codificado.select_dtypes(include=['object']).columns
            # Crear un objeto LabelEncoder para cada columna categórica y transformar los datos
            label_encoders = {col: LabelEncoder() for col in object_columns}
            for col in object_columns:
                df_codificado[col] = label_encoders[col].fit_transform(df_codificado[col])
        else:
            raise ValueError("El parámetro 'metodo' debe ser uno de: 'dummy', 'ohe', 'label'.")
        # Regresar el DataFrame 'df_codificado' completo con las columnas categóricas codificadas
        return df_codificado
        
    @classmethod
    def eliminar_nulos_si(cls, df, p=0.5):
        """
        Elimina las columnas que tienen un porcentaje mayor o igual a p de valores nulos.
        Parameters:
            p (float): Porcentaje límite para considerar una columna con valores nulos.
                       Por defecto es 0.5 (50%).
        """
        nan_percentages = df.isna().mean()
        columns_to_drop = nan_percentages[nan_percentages >= p].index
        aux = df.drop(columns=columns_to_drop)
        return aux

    @classmethod
    def imputar_faltantes(cls, df, metodo="mm", n_neighbors=5):
        """
        Imputa los valores faltantes en un DataFrame utilizando diferentes métodos.
        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene los valores faltantes.
            metodo (str, optional): El método de imputación a utilizar. Opciones válidas: "mm" (por defecto) o "knn".
            n_neighbors (int, optional): El número de vecinos más cercanos a utilizar en el método KNNImputer.
                                         Solo aplicable si el método es "knn".
        Retorna:
            pandas DataFrame: El DataFrame original con los valores faltantes imputados.
        Método "mm" (Mediana/Moda):
            - Imputa el valor de la mediana a los valores faltantes en variables numéricas.
            - Imputa la moda en variables categóricas.
        Método "knn" (K-Nearest Neighbors):
            - Imputa los valores faltantes utilizando KNNImputer, un método basado en vecinos más cercanos.
        Ejemplo de uso:
            df_imputado = Eda.imputar_faltantes(df, metodo="knn", n_neighbors=5)
        """
        if metodo == "mm":
            """
            Imputa el valor de la mediana a los valores nulos en variables numéricas
            y la moda en variables categóricas.
            """
            df_imputed = df.copy(deep=True)
            for col in df_imputed.columns:
                if df_imputed[col].dtype == 'object':
                    # Imputa la moda a los valores nulos en una columna categórica.
                    mode = df_imputed[col].mode()[0]
                    df_imputed[col].fillna(mode, inplace=True)
                else:
                    # Imputa la mediana a los valores nulos en una columna numérica.
                    median = df_imputed[col].median()
                    df_imputed[col].fillna(median, inplace=True)
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
        else:
            raise ValueError("El parámetro 'metodo' debe ser uno de: 'mm', 'knn'.")
        return df_imputed
        
    @classmethod
    def eliminar_unitarios(cls, df):
        """
        Elimina las variables que tienen un solo valor.
        """
        columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
        df_copy = df.drop(columns=columns_to_drop)
        return df_copy
    
    @classmethod
    def mezclar_datos(cls, df):
        """
        Mezcla los datos en el DataFrame de forma aleatoria.
        Parámetros:
            df (pandas DataFrame): El DataFrame con los datos a mezclar.
        Retorna:
            pandas DataFrame: Un nuevo DataFrame con las filas mezcladas de manera aleatoria.
        """
        shuffled_df = df.sample(frac=1, random_state=np.random.randint(1, 1000))
        return shuffled_df
    
    @classmethod
    def estandarizar_variables(cls, df, target:str, metodo="zscore"):
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
        y= df[target]
        aux = df.drop(target, axis=1)
        if metodo == 'zscore':
            scaler = StandardScaler()
        elif metodo == 'minmax':
            scaler = MinMaxScaler()
        elif metodo == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("El parámetro 'metodo' debe ser uno de: 'zscore', 'minmax', 'robust'.")
        aux[aux.select_dtypes(include='number').columns] = scaler.fit_transform(aux.select_dtypes(include='number'))
        aux = pd.concat([aux, y], axis=1)
        return aux
    
    @classmethod
    def balancear_datos(cls, df, target:str):
        """
        Equilibra un conjunto de datos desequilibrado en una tarea 
        de clasificación mediante el método de sobre muestreo.
        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables de entrada y la variable objetivo.
            target (str): El nombre de la columna que contiene la variable objetivo.
        Retorna:
            numpy array, numpy array: El conjunto de datos equilibrado (X_balanced) 
            y las etiquetas de las clases equilibradas (y_balanced).
        Ejemplo:
            # Supongamos que tenemos un DataFrame df con clases desequilibradas
            # y queremos equilibrar las clases utilizando el método de sobre muestreo:
            X_balanced, y_balanced = Eda.balancear_datos(df, 'variable_objetivo')
        """
        X = df.drop(target, axis=1)  # Datos de entrada
        y = LabelEncoder().fit_transform(df[target])  # Etiquetas de las clases (variable objetivo)
        # Realizamos el muestreo de datos utilizando resample
        X_resampled, y_resampled = resample(X[y == 0], y[y == 0], replace=True, n_samples=X[y == 1].shape[0]) # type: ignore
        # Agregamos las muestras de la clase minoritaria al conjunto original
        X_balanced = np.vstack((X, X_resampled))
        y_balanced = np.hstack((y, y_resampled))
        # Convertimos y_balanced en un DataFrame
        y_balanced_df = pd.DataFrame(y_balanced, columns=[target])
        # Concatenamos X_balanced y y_balanced_df horizontalmente
        df_balanced = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), y_balanced_df], axis=1)
        return df_balanced
    
    @classmethod
    def all_eda(cls,df,
                target:str,
                p=0.5,
                imputar=True,
                metodo_imputar="mm",
                n_neighbors=5,
                convertir=True,
                metodo_convertir="ohe",
                estandarizar=False,
                metodo_estandarizar="zscore",
                balancear=False,
                mezclar=False):
        """
        Realiza un Análisis Exploratorio de Datos (EDA) completo en un DataFrame dado.
        Parámetros:
            df (pandas DataFrame): El DataFrame con los datos a analizar.
            target (str): El nombre de la columna que contiene la variable objetivo.
            balancear (bool, optional): Si True, balancea los datos en función de la variable objetivo.
                                        Por defecto es False.
            p (float, optional): Umbral para eliminar columnas con valores faltantes en más de p fracción de filas.
                                 Por defecto es 0.5 (eliminar columnas con más del 50% de valores faltantes).
            imputar (bool, optional): Si True, imputa los valores faltantes en el DataFrame después de eliminar
                                      las columnas con valores faltantes. Por defecto es True.
            metodo_imputar (str, optional): El método de imputación a utilizar si imputar=True. Opciones válidas:
                                            "mm" (por defecto) para imputar la mediana en variables numéricas y la moda en categóricas,
                                            "knn" para utilizar KNNImputer.
            n_neighbors (int, optional): El número de vecinos más cercanos a utilizar en el método KNNImputer.
                                         Solo aplicable si metodo_imputar="knn".
            estandarizar (bool, optional): Si True, estandariza las variables numéricas en el DataFrame después de imputar los valores faltantes.
                                           Por defecto es False.
            metodo_estandarizar (str, optional): El método de estandarización a utilizar si estandarizar=True. 
                                                Opciones válidas:
                                                 "zscore" (por defecto) para estandarización Z-score,
                                                 "minmax" para Min-Max scaling,
                                                 "robust" para Robust scaling.
            mezclar (bool, optional): Si True, mezcla los datos del DataFrame. Por defecto False
        Retorna:
            pandas DataFrame: El DataFrame con los datos limpios y procesados después de aplicar el EDA completo.
        Ejemplo de uso:
            df_cleaned = Eda.all_eda(df, target="target", balancear=True, 
                                    p=0.3, imputar=True, metodo_imputar="knn", 
                                    n_neighbors=5, estandarizar=True, 
                                    metodo_estandarizar="zscore", mezclar=True)
        """
        df_limpio = cls.eliminar_unitarios(df)
        df_limpio = cls.eliminar_nulos_si(df_limpio,p)
        if imputar:
            df_limpio = cls.imputar_faltantes(df_limpio,metodo_imputar,n_neighbors)
        if convertir:
            df_limpio = cls.convertir_a_numericas(df_limpio,target,metodo=metodo_convertir)
        if estandarizar:
            df_limpio = cls.estandarizar_variables(df_limpio,metodo_estandarizar)
        if balancear:
            df_limpio = cls.balancear_datos(df_limpio,target)
        if mezclar:
            df_limpio = cls.mezclar_datos(df_limpio)
        return df_limpio
    
class Graph:

    @classmethod
    def graficos_categoricos(cls, df)->None:
        """
        Crea gráficos de barras horizontales para cada variable categórica en el DataFrame.
        Parámetros:
            df (pandas DataFrame): El DataFrame que contiene las variables categóricas a graficar.
        """
        # Seleccionar columnas categóricas del DataFrame
        categorical_columns = df.select_dtypes('O').columns
        # Calcular el número de columnas categóricas y filas para organizar los gráficos
        num_columns = len(categorical_columns)
        rows = (num_columns + 1) // 2
        # Crear la figura y los ejes de los gráficos
        _, ax = plt.subplots(nrows=rows, ncols=2, figsize=(16, rows * 6))
        ax = ax.flat
        # Generar gráficos de barras horizontales para cada variable categórica
        for i, col in enumerate(categorical_columns):
            df[col].value_counts().plot.barh(ax=ax[i])
            ax[i].set_title(col, fontsize=12, fontweight="bold")
            ax[i].tick_params(labelsize=12)
        # Ajustar el diseño y mostrar los gráficos
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_histogram(cls, df, column:str)->None:
        """
        Genera un histograma interactivo para una columna específica del DataFrame.
        Parameters:
            column (str): Nombre de la columna a visualizar en el histograma.
        """
        fig = px.histogram(df, x=column)
        fig.show()

    @classmethod
    def plot_boxplot(cls, df, x:str, y:str)->None:
        """
        Genera un gráfico de caja interactivo para una variable y en función de otra variable x.
        Parameters:
            x (str): Nombre de la variable independiente en el gráfico de caja.
            y (str): Nombre de la variable dependiente en el gráfico de caja.
        """
        fig = px.box(df, x=x, y=y)
        fig.show()

    @classmethod
    def plot_scatter(cls, df, x:str, y:str)->None:
        """
        Genera un gráfico de dispersión interactivo para dos variables x e y.
        Parameters:
            x (str): Nombre de la variable x en el gráfico de dispersión.
            y (str): Nombre de la variable y en el gráfico de dispersión.
        """
        fig = px.scatter(df, x=x, y=y)
        fig.show()