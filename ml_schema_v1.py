# -*- coding: utf-8 -*-
"""
Autor: ale-uy
Fecha: 27 de Julio de 2023
Archivo: ml_schema_v1.py
Descripción: Automatizar procesos de analisis en datos y machine learning
Licencia: Apache License Version 2.0
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


class eda:
    
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
        ToDo
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
            pass

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
        Genera un histograma para una columna dada.

        Parameters:
            column (str): Nombre de la columna para la cual generar el histograma.
        """
        plt.figure()
        df[column].plot.hist()
        plt.title(f"Histogram of {column}")
        plt.show()

    @classmethod
    def plot_boxplot(cls, df, x:str, y:str):
        """
        Genera un diagrama de caja y bigotes para visualizar la distribución de una variable numérica
        agrupada por una variable categórica.

        Parameters:
            x (str): Nombre de la variable categórica para agrupar.
            y (str): Nombre de la variable numérica a visualizar.
        """
        plt.figure()
        sns.boxplot(x=x, y=y, data=df)
        plt.title(f"Boxplot of {y} grouped by {x}")
        plt.show()

    @classmethod
    def plot_scatter(cls, df, x:str, y:str):
        """
        Genera un gráfico de dispersión para visualizar la relación entre dos variables numéricas.

        Parameters:
            x (str): Nombre de la variable en el eje x.
            y (str): Nombre de la variable en el eje y.
        """
        plt.figure()
        plt.scatter(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"Scatter plot of {x} and {y}")
        plt.show()

    @classmethod
    def plot_histogram_interactive(cls, df, column:str):
        """
        Genera un histograma interactivo para una columna específica del DataFrame.

        Parameters:
            column (str): Nombre de la columna a visualizar en el histograma.
        """
        fig = px.histogram(df, x=column)
        fig.show()

    @classmethod
    def plot_boxplot_interactive(cls, df, x:str, y:str):
        """
        Genera un gráfico de caja interactivo para una variable y en función de otra variable x.

        Parameters:
            x (str): Nombre de la variable independiente en el gráfico de caja.
            y (str): Nombre de la variable dependiente en el gráfico de caja.
        """
        fig = px.box(df, x=x, y=y)
        fig.show()

    @classmethod
    def plot_scatter_interactive(cls, df, x:str, y:str):
        """
        Genera un gráfico de dispersión interactivo para dos variables x e y.

        Parameters:
            x (str): Nombre de la variable x en el gráfico de dispersión.
            y (str): Nombre de la variable y en el gráfico de dispersión.
        """
        fig = px.scatter(df, x=x, y=y)
        fig.show()
