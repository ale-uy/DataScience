# -*- coding: utf-8 -*-
"""
Autor: ale-uy
Fecha: 08/2023
Actualizado: 10/2023
Version: v0.1
Archivo: dl.py
Descripción: Aplicar modelos de redes neuronales.
Licencia: Apache License Version 2.0
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Tools:

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
                y = pd.Series(y_encoded, name=target)

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test
    

class DL(Tools):

    @classmethod
    def modelo_FNN(cls,
                   df,
                   target,
                   tipo_problema,
                   num_unidades=16,
                   num_capas=1,
                   num_clases=1,
                   tasa_dropout=0.2,
                   optimizador='adam',
                   tasa_aprendizaje=0.01,
                   activacion='relu',
                   num_epochs=10,
                   batch_size=32):
        """
        Crea un modelo de red neuronal feedforward (FNN) personalizable.
        Recomendado para problemas de regresion o Clasificacion.

        Args:
            df (pandas DataFrame): El DataFrame que contiene los datos.
            target (str): El nombre de la columna objetivo.
            tipo_problema (str): Puede ser 'clasificacion', 'regresion' o 'multiclase'.
            num_unidades (int, optional): Número de unidades en cada capa oculta. Default es 16.
            num_capas (int, optional): Número de capas ocultas. Default es 1.
            num_clases (int): Número de clases de salida. Defecto es 1
            tasa_dropout (float, optional): Tasa de dropout para las capas de dropout. Default es 0.2.
            optimizador (str, optional): Optimizador a utilizar: 'adam', 'rmsprop', 'sgd'... Default es 'adam'.
            tasa_aprendizaje (float, optional): Tasa de aprendizaje para el optimizador. Default es 0.01.
            activacion (str, optional): Función de activación 'relu', 'softmax', 'sigmoid'... Default 'relu'.
            num_epochs (int, optional): Número de epochs para entrenar el modelo. Default es 10.
            batch_size (int, optional): Tamaño del batch para el entrenamiento. Default es 32.

        Cargar Modelo:
            # Recrea exactamente el mismo modelo solo desde el archivo
            my_model = tf.keras.models.load_model('modelo_fnn.h5')
            # Usarlo para predicción
            prediction = my_model.predict(input_data)
        
        Returns:
            tensorflow.keras.models.Sequential: Modelo de red neuronal feedforward (FNN).
        """
        model = Sequential()

        # Capas ocultas FNN
        for _ in range(num_capas):
            model.add(Dense(num_unidades, activation=activacion))
            model.add(Dropout(tasa_dropout))

        if tipo_problema == 'regresion':
            # Caso de regresión (una sola clase)
            activacion_salida = 'linear'
            loss, metric = 'mean_squared_error', 'mean_squared_error'
        elif tipo_problema == 'clasificacion':
            # Caso de clasificación binaria (dos clases)
            activacion_salida = 'sigmoid'
            loss, metric = 'binary_crossentropy', 'accuracy'
        elif tipo_problema == 'multiclase':
            # Caso de clasificación multiclase (más de dos clases)
            activacion_salida = 'softmax'
            loss, metric = 'categorical_crossentropy', 'accuracy'
        else:
            raise ValueError('opciones en tipo_problema: "regresion", "clasificacion" o "multiclase"')

        model.add(Dense(num_clases, activation=activacion_salida))

        optimizer = tf.keras.optimizers.get(optimizador, learning_rate=tasa_aprendizaje)
        
        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

        # Entrenamiento del modelo
        X_train, X_test, y_train, y_test = cls.dividir_y_convertir_datos(df, target)
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

        # Evaluar la eficacia del modelo en datos de prueba
        eval_result = model.evaluate(X_test, y_test)
        print(f"\nEficacia del modelo en datos de prueba: \nLoss = {eval_result[0]:.3f} \n{metric.capitalize()} = {eval_result[1]:.3f}")

        # Guardar el Modelo
        model.save('modelo_fnn.h5')
        print('\nEl modelo se ah guardado como "modelo_fnn.h5"')

        return model
    
