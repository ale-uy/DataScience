# -*- coding: utf-8 -*-
"""
Autor: ale-uy
Fecha: 27 de Julio de 2023
Archivo: ml_schema.py
Descripción: Automatizar machine learning
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class preproceso:

    @classmethod
    def one_hot_encode(cls, df, categorical_vars:list):
        """
        Aplica OneHotEncoding a las variables categóricas seleccionadas en un DataFrame.

        Parámetros:
        df (pandas DataFrame): El DataFrame que contiene las variables categóricas.
        categorical_vars (list): Una lista de nombres de las columnas categóricas a codificar.

        Retorna:
        pandas DataFrame: El DataFrame original con las columnas categóricas codificadas mediante OneHotEncoding.
        """

        # Crear el codificador OneHotEncoder
        encoder = OneHotEncoder(sparse=False, drop='first')
        # Obtener las columnas categóricas a codificar
        categorical_data = df[categorical_vars]
        # Realizar OneHotEncoding y reemplazar las columnas categóricas originales
        encoded_data = encoder.fit_transform(categorical_data)
        df.drop(columns=categorical_vars, inplace=True)
        # Crear nombres para las nuevas columnas
        new_columns = encoder.get_feature_names_out(input_features=categorical_vars)
        # Crear un DataFrame con las nuevas columnas y unirlo al DataFrame original
        df = pd.concat([df, pd.DataFrame(encoded_data, columns=new_columns)], axis=1)

    @classmethod
    def estandarizar_variables(cls, df, variables_a_transformar:list):
        """
        Estandariza las variables numéricas seleccionadas en el DataFrame utilizando StandardScaler de sklearn.
        Pasar una lista de nombres de variables a transformar.
        NOTA: No hacerlo con: Árboles de decisión, Bosques aleatorios, ExtraTrees, k-Nearest Neighbors,
        y metodos de ensamble en general como LightGBM, XGBoost, CatBoost, AdaBoost...
        """

        # Filtrar las columnas numéricas seleccionadas
        numeric_columns = df[variables_a_transformar].select_dtypes(include='number').columns
        # Escala las variables seleccionadas
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


class ml:
    def __init__(self, df):
        """
        Inicializa una instancia de la clase ml.

        Parameters:
            df (DataFrame): El DataFrame sobre el cual realizar el aprendizaje.
        """
        self.df = df

    