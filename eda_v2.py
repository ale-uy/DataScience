# -*- coding: utf-8 -*-
"""
Autor: ale-uy
Fecha: 2 de mayo de 2023
Archivo: eda_v2.py
Descripción: Análisis Exploratorio de Datos (EDA en ingles)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


class Eda:
    def __init__(self, df):
        """
        Inicializa una instancia de la clase Eda.

        Parameters:
            df (DataFrame): El DataFrame sobre el cual realizar el análisis exploratorio de datos.
            df_copy (DataFrame): Copia del DataFrame para cuando apliquemos técnicas de ML por ejemplo.
        """
        self.df = df
        self.df_copy = pd.DataFrame()

    def analizar_nulos(self):
        """
        Devuelve el porcentaje de valores nulos en el total de datos para cada columna.

        Returns:
            Series: Una serie que contiene los porcentajes de valores nulos para cada columna,
                    ordenados de mayor a menor.
        """
        return self.df.isna().sum().sort_values(ascending=False) / self.df.shape[0] * 100

    def eliminar_nulos_si(self, p=0.5):
        """
        Elimina las columnas que tienen un porcentaje mayor o igual a p de valores nulos.

        Parameters:
            p (float): Porcentaje límite para considerar una columna con valores nulos.
                       Por defecto es 0.5 (50%).
        """
        nan_percentages = self.df.isna().mean()
        mask = nan_percentages >= p
        self.df = self.df.loc[:, ~mask]

    def graficos_categoricos(self):
        """
        Crea gráficos de barras horizontales para cada variable categórica en el DataFrame.
        """
        categorical_columns = self.df.select_dtypes('O').columns
        num_columns = len(categorical_columns)
        rows = (num_columns + 1) // 2
        _, ax = plt.subplots(nrows=rows, ncols=2, figsize=(16, rows * 6))
        ax = ax.flat
        for i, col in enumerate(categorical_columns):
            self._plot_categorical_barh(col, ax[i])
        plt.tight_layout()
        plt.show()

    def _plot_categorical_barh(self, column, ax):
        """
        Crea un gráfico de barras horizontales para una variable categórica.

        Parameters:
            column (str): Nombre de la columna categórica.
            ax: Eje donde se muestra el gráfico.
        """
        self.df[column].value_counts().plot.barh(ax=ax)
        ax.set_title(column, fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=12)

    def imputar_moda_mediana(self):
        """
        Imputa el valor de la mediana a los valores nulos en variables numéricas
        y la moda en variables categóricas.
        """
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self._impute_categorical_mode(col)
            else:
                self._impute_numeric_median(col)

    def _impute_categorical_mode(self, column):
        """
        Imputa la moda a los valores nulos en una columna categórica.

        Parameters:
            column (str): Nombre de la columna categórica.
        """
        mode = self.df[column].mode()[0]
        self.df[column].fillna(mode, inplace=True)

    def _impute_numeric_median(self, column):
        """
        Imputa la mediana a los valores nulos en una columna numérica.

        Parameters:
            column (str): Nombre de la columna numérica.
        """
        median = self.df[column].median()
        self.df[column].fillna(median, inplace=True)

    def eliminar_unitarios(self):
        """
        Elimina las variables que tienen un solo valor.
        """
        columns_to_drop = [col for col in self.df.columns if self.df[col].nunique() == 1]
        self.df.drop(columns_to_drop, axis=1, inplace=True)

    def estadisticos_numericos(self):
        """
        Genera datos estadísticos de las variables numéricas en el DataFrame.

        Returns:
            DataFrame: Un DataFrame que contiene los estadísticos de las variables numéricas,
                       incluyendo count, mean, std, min, 25%, 50%, 75% y max.
        """
        return self.df.select_dtypes('number').describe().T

    def shuffle_data(self):
        """
        Mezcla los datos en el DataFrame de forma aleatoria.
        """
        self.df = self.df.iloc[np.random.permutation(len(self.df))]

    def create_copy(self):
        """
        Crea una copia del DataFrame original `df` y lo asigna al DataFrame `df_hot`.

        El DataFrame `df_copy` lo utilizamos para resguardar el DataFrame orginal.
        """
        self.df_copy = self.df.copy(deep=True)


    def create_dummies(self, drop_first=False):
        """
        Realiza la codificación one-hot/dummies de las variables categóricas en el DataFrame.
        Con drop_first = True elimina la primer dummy. Por defecto es False (one-hot-encoding)
        """
        #categorical_columns = self.df.select_dtypes('object').columns
        #prefixes = {k:v for (k,v) in zip(categorical_columns, categorical_columns)}
        self.df = pd.get_dummies(self.df, drop_first=drop_first)

    def plot_histogram(self, column):
        """
        Genera un histograma para una columna dada.

        Parameters:
            column (str): Nombre de la columna para la cual generar el histograma.
        """
        plt.figure()
        self.df[column].plot.hist()
        plt.title(f"Histogram of {column}")
        plt.show()

    def plot_boxplot(self, x, y):
        """
        Genera un diagrama de caja y bigotes para visualizar la distribución de una variable numérica
        agrupada por una variable categórica.

        Parameters:
            x (str): Nombre de la variable categórica para agrupar.
            y (str): Nombre de la variable numérica a visualizar.
        """
        plt.figure()
        sns.boxplot(x=x, y=y, data=self.df)
        plt.title(f"Boxplot of {y} grouped by {x}")
        plt.show()

    def plot_scatter(self, x, y):
        """
        Genera un gráfico de dispersión para visualizar la relación entre dos variables numéricas.

        Parameters:
            x (str): Nombre de la variable en el eje x.
            y (str): Nombre de la variable en el eje y.
        """
        plt.figure()
        plt.scatter(self.df[x], self.df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"Scatter plot of {x} and {y}")
        plt.show()

    def plot_histogram_interactive(self, column):
        """
        Genera un histograma interactivo para una columna específica del DataFrame.

        Parameters:
            column (str): Nombre de la columna a visualizar en el histograma.
        """
        fig = px.histogram(self.df, x=column)
        fig.show()

    def plot_boxplot_interactive(self, x, y):
        """
        Genera un gráfico de caja interactivo para una variable y en función de otra variable x.

        Parameters:
            x (str): Nombre de la variable independiente en el gráfico de caja.
            y (str): Nombre de la variable dependiente en el gráfico de caja.
        """
        fig = px.box(self.df, x=x, y=y)
        fig.show()

    def plot_scatter_interactive(self, x, y):
        """
        Genera un gráfico de dispersión interactivo para dos variables x e y.

        Parameters:
            x (str): Nombre de la variable x en el gráfico de dispersión.
            y (str): Nombre de la variable y en el gráfico de dispersión.
        """
        fig = px.scatter(self.df, x=x, y=y)
        fig.show()