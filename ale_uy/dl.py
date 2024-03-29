# -*- coding: utf-8 -*-
"""
Author: ale-uy
Date: 08/2023
Updated: 10/2023
Version: v1
File: dl.py
Description: Apply neural network models.
License: Apache License Version 2.0
"""

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from sklearn.neural_network import MLPClassifier, MLPRegressor
from ml import Tools


class DL(Tools):

    @classmethod
    def model_FNN(cls,
                   df,
                   target,
                   problem_type,
                   num_units=16,
                   num_layers=1,
                   num_class=1,
                   droput_rate=0.2,
                   optimizer='adam',
                   learning_rate=0.01,
                   activation='relu',
                   num_epochs=10,
                   grid=False,
                   batch_size=32,
                   max_trials=10,
                   test_size=0.2,
                   random_state=1):
        
        """
        Creates a customizable Feedforward Neural Network (FNN) model.
        Recommended for regression or classification problems.

        Args:
            df (pandas DataFrame): The DataFrame containing the data.
            target (str): The name of the target column.
            problem_type (str): Can be 'classification' or 'regression'.
            num_units (int, optional): Number of units in each hidden layer. Default is 16.
            num_layers (int, optional): Number of hidden layers. Default is 1.
            num_class (int): Number of output classes. Default is 1.
            dropout_rate (float, optional): Dropout rate for dropout layers. Default is 0.2.
            optimizer (str, optional): Optimizer to use: 'adam', 'rmsprop', 'sgd', etc. Default is 'adam'.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
            activation (str, optional): Activation function, e.g., 'relu', 'softmax', 'sigmoid', etc. Default is 'relu'.
            num_epochs (int, optional): Number of epochs to train the model. Default is 10.
            grid (bool, optional): Search for hyperparameters. Default is False.
            batch_size (int, optional): Batch size for training. Default is 32.
            max_trials (int, optional): If grid=True, use to search best params this number time. Default 10.
            test_size (float, opt): The sample size for the test set, default is 0.2.
            random_state (int): Seed to use for data splitting, defaults to a random number.

        Load Model:
            # Recreate the exact same model from the file
            my_model = tf.keras.models.load_model('fnn_model.h5')
            # Use it for prediction
            prediction = my_model.predict(input_data)

        Returns:
            tensorflow.keras.models.Sequential: Feedforward Neural Network (FNN) model.
        """

        if problem_type == 'classification' and df['target'].nunique() > 2:
                raise NotImplementedError('Sorry, multiclass classification not found!')

        model = Sequential()

        if problem_type == 'regression':
            # Regression case
            activation_out = 'linear'
            loss, metric = 'mean_squared_error', 'mean_squared_error'
        elif problem_type == 'classification':
            # Binary classification case (two class)
            activation_out = 'sigmoid'
            loss, metric = 'binary_crossentropy', 'accuracy'
        elif problem_type == 'multiclass':
            # Multiclass classification case (many classes)
            activation_out = 'softmax'
            loss, metric = 'categorical_crossentropy', 'accuracy'
        else:
            raise ValueError('Options in problem_type: "regression" or "classification"')

        if grid:
            
            def build_model(hp):

                # Create a new Sequential model in each iteration
                model = Sequential()

                # Hidden Layers
                for i in range(num_layers):
                    model.add(Dense(units=hp.Int('units_' + str(i), min_value=8, max_value=128, step=8), 
                                    activation=hp.Choice('activation_' + str(i), values=['relu', 'tanh', 'sigmoid', 'softmax']),
                                    name='dense_' + str(i)))  # Assign unique name for each dense layer
                    model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))

                model.add(Dense(num_class, activation=activation_out))

                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=hp.Float('learning_rate', min_value=0.0001, max_value=0.1, sampling='log')
                    )

                model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

                return model

            # Defines the hyperparameter search space
            tuner = RandomSearch(
                build_model,
                objective='val_loss',
                max_trials=max_trials,  # Number of configurations to test
                directory='tuner_dir',
                project_name='my_tuner')

            # Separate the target variable 'y' from the rest of the variables 'X'
            X = df.drop(columns=[target])
            y = df[target]
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Train the tuner to find the best hyperparameters
            tuner.search(X_train, y_train, validation_data=(X_test, y_test), 
                         epochs=num_epochs, batch_size=batch_size)

            # Get the model with the best hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps)

        else:    
            # Hidden layers
            for _ in range(num_layers):
                model.add(Dense(num_units, activation=activation))
                model.add(Dropout(droput_rate))

            model.add(Dense(num_class, activation=activation_out))

            optimizer = tf.keras.optimizers.get(optimizer, learning_rate=learning_rate)

            model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

            # Model training
            X_train, X_test, y_train, y_test = Tools.split_and_convert_data(df, target)

        model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

        # Evaluate model effectiveness on test data
        eval_result = model.evaluate(X_test, y_test)
        print(f"\nModel efficiency on test data: \nLoss = {eval_result[0]:.3f} \nScore = {eval_result[1]:.3f}")

        # Save the Model
        model.save('model_fnn.h5')
        print('\nThe model has been saved as "model_fnn.h5"')
        print('\nTo load the model from the .h5 file, use: model = tf.keras.models.load_model("model_fnn.h5")')

        return model
    
    @classmethod
    def model_ANN(cls,
                   df,
                   target,
                   problem_type,
                   learning_rate=0.001,
                   num_units=100,
                   num_layers=1,
                   optimizer='adam',
                   activation='relu',
                   num_epochs=10,
                   alpha=0.0001,
                   momentum=0.9,
                   grid=False,
                   cv=5,
                   n_iter=10,
                   batch_size='auto',
                   test_size=0.2,
                   random_state=1):
        """
        Create a customizable Artificial Neural Network (ANN) model using scikit-learn.

        Args:
            df (pandas DataFrame): The DataFrame containing the data.
            target (str): The name of the target column.
            problem_type (str): Can be 'classification' or 'regression'.
            alpha (float, optional): L2 regularization term. Default is 0.0001.
            num_units (int, optional): Number of units in each hidden layer. Default is 100.
            num_layers (int, optional): Number of hidden layers. Default is 1.
            optimizer (str, optional): Optimizer to use: 'adam', 'sgd', 'lbfgs', etc. Default is 'adam'.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
            activation (str, optional): Activation function, e.g., 'logistic', 'tanh', 'relu', etc. Default is 'relu'.
            num_epochs (int, optional): Number of epochs to train the model. Default is 10.
            momentum (float, optional): Gradient descent update [0 and 1]. Only with solver='sgd'.
            grid (bool, optional): Enable hyperparameter tuning. Default is False.
            cv (int, optional): Number of cross-validation partitions. Default is 5 if grid=True.
            n_iter (int, optional): Number of hyperparameter combinations to try. Default is 10 if grid=True.
            batch_size (int, optional): Batch size for training. Default is 'auto'.
            test_size (float, opt): The sample size for the test set, default is 0.2.
            random_state (int): Seed to use for data splitting, defaults to a random number.

        Returns:
            sklearn.neural_network.MLPClassifier or sklearn.neural_network.MLPRegressor: 
                Artificial Neural Network (ANN) model.
        """

        # Separate the target variable 'y' from the rest of the variables 'X'
        X = df.drop(columns=[target])
        y = df[target]
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if problem_type == 'classification' and y_train.nunique() > 2:
                raise NotImplementedError('Sorry, multiclass classification not found!')

        if grid:
            param_grid = {
                'hidden_layer_sizes': [(16,), (32,), (64,), (16, 16), (32, 32)],
                'activation': ['relu', 'tanh', 'logistic', 'softmax'],
                'learning_rate_init': [0.001, 0.01, 0.1, 1.0],
                'solver': ['adam', 'sgd'],
                'max_iter': [50, 100, 200],
            }
            model = MLPRegressor() if problem_type=='regression' else MLPClassifier()
            score = 'neg_mean_squared_error' if problem_type=='regression' else 'accuracy'

            best_params = Tools._random_search(model, param_grid, X_train, y_train, score, cv, n_iter)

            # Display the values selected by GridSearch
            print(pd.DataFrame(best_params))
            # Use the best-found hyperparameters
            params = best_params['best_params']
        else:
            params = {
                'hidden_layer_sizes': (num_units,num_layers),
                'activation': activation,
                'learning_rate_init': learning_rate,
                'solver': optimizer,
                'max_iter': num_epochs,
            }

        # Create the ANN model
        if problem_type == 'regression':
            model = MLPRegressor(**params,
                                 alpha=alpha,
                                 batch_size=batch_size,
                                 early_stopping=True,
                                 momentum=momentum,
                                 random_state=1)
            
        elif problem_type == 'classification':
            model = MLPClassifier(**params,
                                 alpha=alpha,
                                 batch_size=batch_size,
                                 early_stopping=True,
                                 momentum=momentum,
                                 random_state=1)
        else:
            raise ValueError('The problem_type must be "regression" or "classification".')

        # Model training
        model.fit(X_train, y_train)

        # Evaluate the model on test data
        accuracy = model.score(X_test, y_test)
        print(f"Model efficiency on test data: {accuracy:.3f}")

        # Save the Model
        joblib.dump(model, 'model_ann.jolib')
        print('\nThe model has been saved as "model_ann.joblib"')
        print('\nTo load the model from the .joblib file, use: model = joblib.load("model_ann.jolib")')

        return model


class Beta(Tools):

    @classmethod
    def model_RNN(cls, 
                  df, 
                  target, 
                  input_shape, 
                  problem_type, 
                  num_units=16, 
                  num_class=1, 
                  use_gru_model=False, 
                  num_layers=1, 
                  dropout_rate=0.2, 
                  optimizer='adam', 
                  learning_rate=0.01, 
                  activation='relu', 
                  num_epochs=10, 
                  batch_size=32,
                  test_size=0.2,
                  random_state=1):
        """
        Create a customizable Recurrent Neural Network (RNN) model. Can be used with LSTM or GRU. Recommended for time series data.

        Args:
            df (pandas DataFrame): The DataFrame containing the data.
            target (str): The name of the target column.
            input_shape (tuple): Input dimensions as a tuple (timesteps, features).
            problem_type (str): Can be 'classification', 'regression', or 'multiclass'.
            num_units (int, optional): Number of units in each RNN layer. Default is 16.
            num_class (int, optional): Number of output classes. Default is 1.
            use_gru_model (bool, optional): The default model is 'LSTM', set to True to use 'GRU'.
            num_layers (int, optional): Number of stacked RNN layers. Default is 1.
            dropout_rate (float, optional): Dropout rate for dropout layers. Default is 0.2.
            optimizer (str, optional): Optimizer to use: 'adam', 'rmsprop', 'sgd', etc. Default is 'adam'.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
            activation (str, optional): Activation function 'relu', 'softmax', 'sigmoid', etc. Default 'relu'.
            num_epochs (int, optional): Number of epochs for model training. Default is 10.
            batch_size (int, optional): Batch size for training. Default is 32.
            test_size (float, opt): The sample size for the test set, default is 0.2.
            random_state (int): Seed to use for data splitting, defaults to a random number.

        Model Loading:
            # Recreate the exact same model from the file
            my_model = tf.keras.models.load_model('rnn_model.h5')
            # Use it for prediction
            prediction = my_model.predict(input_data)

        Returns:
            tensorflow.keras.models.Sequential: RNN model (LSTM or GRU).
        """
        model = Sequential()

        # RNN Input Layer
        if use_gru_model:
            model.add(GRU(num_units, input_shape=input_shape, return_sequences=True))
        else:
            model.add(LSTM(num_units, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(dropout_rate))

        # Additional LSTM Layers (if num_layers > 1)
        for _ in range(num_layers - 1):
            if use_gru_model:
                model.add(GRU(num_units, return_sequences=True))
            else:
                model.add(LSTM(num_units, return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(Dense(num_units, activation=activation))  # Activation function in hidden layers

        # Output Layer
        if use_gru_model:
            model.add(GRU(num_units))
        else:
            model.add(LSTM(num_units))
        model.add(Dropout(dropout_rate))

        if problem_type == 'regression':
            # Regression case (single class)
            output_activation = 'linear'
            loss, metric = 'mean_squared_error', 'mean_squared_error'
        elif problem_type == 'classification':
            # Binary classification case (two classes)
            output_activation = 'sigmoid'
            loss, metric = 'binary_crossentropy', 'accuracy'
        elif problem_type == 'multiclass':
            # Multiclass classification case (more than two classes)
            output_activation = 'softmax'
            loss, metric = 'categorical_crossentropy', 'accuracy'
        else:
            raise ValueError('problem_type options: "regression", "classification", or "multiclass"')

        model.add(Dense(num_class, activation=output_activation))

        optimizer = tf.keras.optimizers.get(optimizer, learning_rate=learning_rate)

        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

        # Separate the target variable 'y' from the rest of the variables 'X'
        X = df.drop(columns=[target])
        y = df[target]
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Model Training
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

        # Evaluate Model Effectiveness on Test Data
        eval_result = model.evaluate(X_test, y_test)
        print(f"\nModel Effectiveness on Test Data: \nLoss = {eval_result[0]:.3f} \n{metric.capitalize()} = {eval_result[1]:.3f}")

        # Save the Model
        model.save('rnn_model.h5')
        print('\nThe model has been saved as "rnn_model.h5"')

        return model
    
