# -*- coding: utf-8 -*-
"""
Author: ale-uy
Date: 07/2023
Updated: 10/2023
Version: v2
File: ml.py
Description: Automate machine learning processes
License: Apache License Version 2.0
"""

import joblib
import warnings
import optuna
import xgboost as xgb
import catboost as cb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, \
    precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_squared_log_error, silhouette_score
from sklearn.cluster import KMeans, DBSCAN



pd.set_option('display.max_colwidth', None) # Display full cell width in the DataFrame
warnings.filterwarnings("ignore")


class Tools:

    @classmethod
    def feature_importance(cls, df: pd.DataFrame, target: str, n_estimators=100, save_model=False, bottom=0,
                          random_state=np.random.randint(1, 1000), cv=5, model_filename="Random_Forest",
                          problem_type=None, eliminate=False, threshold=0.0001):
        """
        Calculates the importance of variables based on their contribution to prediction using RandomForest.

        Parameters:
            df (pandas DataFrame): The DataFrame containing input variables and the target variable.
            target (str): The name of the column containing the target variable.
            n_estimators (int): Number of trees to be used for classification, 100 by default.
            random_state (int): Seed to use, by default, it's a random number.
            cv (int): Number of cross-validation folds, 5 by default.
            bottom (int): How many variables from the bottom to display (all by default).
            save_model (bool): True to save the model (False by default).
            eliminate (bool): Whether to eliminate less important variables. By default, it's False.
            problem_type (str): 'classification' or 'regression' (None by default).
            threshold (float): Threshold value determining the minimum importance required to keep a variable.
                              By default, it's 0.005.

        Returns:
            pandas DataFrame: A DataFrame containing the importance ranking of each variable.
            float: Model performance measured by the corresponding metric on the test set.
        """
        # Separate the target variable 'y' from the rest of the variables 'X'
        X = df.drop(columns=[target])
        y = df[target]
        if not np.issubdtype(y.dtype, np.number):
            # If it's not numeric, convert it using LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            y = pd.Series(y_encoded, name=target)

        # Create and train a RandomForest model
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state) \
            if problem_type == 'classification' else \
                RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

        rf_model.fit(X, y)

        # Calculate model accuracy using cross-validation
        scoring = 'accuracy' if problem_type == 'classification' else 'neg_mean_squared_error'
        score = cross_val_score(rf_model, X, y, cv=cv, scoring=scoring).mean()

        # Create a DataFrame with variables and their importance
        importance_df = pd.DataFrame({'Variable': X.columns, 'Importance': rf_model.feature_importances_})
        # Sort the DataFrame by importance in descending order
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        if save_model and model_filename:
            # Save the trained model to disk
            joblib.dump(rf_model, f'{model_filename}.pkl')

        if eliminate:
            # Get the variables that exceed the importance threshold
            important_variables = importance_df[importance_df['Importance'] >= threshold]['Variable']
            # Filter the original DataFrame keeping only the important variables
            filtered_df = df[df.columns[df.columns.isin(important_variables) | df.columns.isin([target])]]
            return filtered_df
        else:
            print(f'MSE of the RF model: {score}')
            print()
            print(f'The least important variables: ')
            print(f'{importance_df[-bottom:]}')

    @classmethod
    def split_and_convert_data(cls, df: pd.DataFrame, target: str, test_size=0.2,
                               random_state=np.random.randint(1, 1000),
                               encode_categorical=False) -> tuple:
        """
        Splits the data into training and test sets and optionally encodes categorical variables.

        Parameters:
            df (pandas DataFrame): The DataFrame containing the data.
            target (str): The name of the target column.
            test_size (float): The size of the test set. Default is 0.2.
            random_state (int): The random seed for data splitting. Default is a random value.
            encode_categorical (bool): Indicates whether categorical variables should be automatically encoded. Default is False.

        Returns:
            tuple: A tuple containing the training and test sets in the order:
            (X_train, X_test, y_train, y_test).
        """
        # Separate the target variable 'y' from the rest of the variables 'X'
        X = df.drop(columns=[target])
        y = df[target]

        # Automatically encode categorical variables using pd.Categorical
        if encode_categorical:
            categorical_columns = X.select_dtypes(include=['object']).columns
            label_encoder = LabelEncoder()
            for col in categorical_columns:
                X[col] = label_encoder.fit_transform(X[col])
            # Check if the target variable is numeric
            if not np.issubdtype(y.dtype, np.number):
                # If it's not numeric, convert it using LabelEncoder
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                y = pd.Series(y_encoded, name=target)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    @classmethod
    def _metrics(cls, y_test, y_pred, metric_type=None):
        """
        Calculates appropriate metrics based on the data types in y_test and y_pred.

        For classification values:
            - Accuracy: Proportion of correctly classified samples.
            - Precision: Proportion of true positive samples among all samples classified as positive.
            - Recall: Proportion of true positive samples among all true positive samples.
            - F1-score: Harmonic mean of precision and recall. Useful when there is class imbalance.
            - AUC-ROC: Area under the ROC curve, which measures the model's discrimination ability.

        For regression values:
            - Mean Squared Error (MSE): Mean squared error between predictions and true values.
            - R-squared (R^2): Coefficient of determination indicating the proportion of total variance of the dependent variable explained by the model.

        Parameters:
            y_test (array-like): True values of the target variable (ground truth).
            y_pred (array-like): Predicted values by the model.
            metric_type (str): You can manually choose whether it's 'classification' or 'regression'.

        Returns:
            pandas DataFrame: A DataFrame containing the metrics and their respective values, along with a brief explanation for each metric.
        """

        if metric_type == "classification":
            # Classification values
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            if len(set(y_pred)) > 2: 
                roc_auc = 'Not Implemented for multiclass'
            else:
                roc_auc = roc_auc_score(y_test, y_pred)

            metric_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'],
                'Value': [accuracy, precision, recall, f1, roc_auc]
            })

        elif metric_type == "regression":
            # Regression values
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            try:
                rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
            except Exception:
                rmsle = 'N/A'
            r2_adj = r2_score(y_test, y_pred, multioutput='variance_weighted')
            rmse = np.sqrt(mse)

            metric_df = pd.DataFrame({
                'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)',
                           'Mean Absolute Error (MAE)', 'Root Mean Squared Logarithmic Error (RMSLE)',
                           'Adjusted R-squared (R^2 adjusted)'],
                'Value': [mse, rmse, mae, rmsle, r2_adj]
            })

        else:
            raise ValueError("The values for 'metric_type' must be 'classification' or 'regression'.")

        return metric_df

    @classmethod
    def _grid_search(cls, model, param_grid, X_train, y_train, scoring='accuracy', cv=5):
        """
        Performs hyperparameter tuning using GridSearchCV.

        Parameters:
            model: The estimator of the model you want to tune.
            param_grid: A dictionary with hyperparameters and their possible values.
            X_train: Training feature set.
            y_train: Labels of the training set.
            scoring: The evaluation metric. Default is 'accuracy'.
            cv: Number of cross-validation partitions. Default is 5.

        Returns:
            dict: A dictionary containing the best-found hyperparameters and the best score.
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
    def _optuna_search(cls, model, param_space, X_train, y_train, scoring='accuracy', n_iter=10, cv=5):
        """
        Performs hyperparameter tuning using Optuna.

        Parameters:
            model: The estimator of the model you want to tune.
            param_space: A dictionary with hyperparameter names and their search spaces.
            X_train: Training feature set.
            y_train: Labels of the training set.
            scoring: The evaluation metric. Default is 'accuracy'.
            n_iter: Number of Optuna trials. Default is 10.
            cv: Number of cross-validation partitions. Default is 5.

        Returns:
            dict: A dictionary containing the best-found hyperparameters and the best score.
        """

        def objective(trial):
            # Define hyperparameters to be optimized
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    lower, upper = param_range
                    if isinstance(lower, int) and isinstance(upper, int):
                        # If it is a range of integers, use suggest_int
                        params[param_name] = trial.suggest_int(param_name, lower, upper)
                    elif isinstance(lower, float) and isinstance(upper, float):
                        # If it is a range of floats, use suggest_uniform
                        params[param_name] = trial.suggest_uniform(param_name, lower, upper)
                    elif all(isinstance(val, str) for val in param_range):
                        # If they are categorical values, use suggest_categorical
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                    else:
                        raise ValueError(f"Unsupported range type for hyperparameter {param_name}")
                else:
                    raise ValueError(f"Invalid range for hyperparameter {param_name}")

            # Set the hyperparameters in the model
            model.set_params(**params)

            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            score = scores.mean()

            return score

        # Create an Optuna study and optimize hyperparameters
        study = optuna.create_study(direction='maximize')  # Or 'minimize' if it is a minimization problem
        study.optimize(objective, n_trials=n_iter)  # Number of test iterations

        # Gets the best combination of hyperparameters found
        best_params = study.best_params
        best_score = study.best_value

        result = {
            'best_params': best_params,
            'best_score': best_score
        }

        return result

    @classmethod
    def _random_search(cls, model, param_dist, X_train, y_train, scoring='accuracy', cv=5, n_iter=10):
        """
        Performs hyperparameter tuning using RandomizedSearchCV.

        Parameters:
            model: The estimator of the model you want to tune.
            param_dist: A dictionary with hyperparameters and their distributions for sampling.
            X_train: Training feature set.
            y_train: Labels of the training set.
            scoring: The evaluation metric. Default is 'accuracy'.
            cv: Number of cross-validation partitions. Default is 5.
            n_iter: Number of hyperparameter combinations to try. Default is 10.

        Returns:
            dict: A dictionary containing the best-found hyperparameters and the best score.
        """
        random_search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_dist, 
            scoring=scoring, 
            cv=cv, 
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
    def generate_clusters(cls, df: pd.DataFrame, k=5, eps=0.5, min_samples=5, dbscan=False, plot=False, random_state=np.random.randint(1,1000)):
        """
        Applies the K-Means or DBSCAN algorithm to a DataFrame and returns a series
        with the cluster number to which each observation belongs.

        :param df: DataFrame with the data for analysis.
        :param k: Number of clusters to generate. Default is 5 (recommended to use Graphs.plot_cluster(df)).
        :param eps: Determines the search radius around each point in feature space.
        :param min_samples: Specifies the minimum number of points within radius eps
            for a point to be considered a core point.
        :param dbscan: If True, applies DBSCAN clustering. If False (default), applies K-Means clustering.
        :param plot: If True, generates a plot of the clusters.
        :param random_state (opt): Seed to use, random by default.

        :return: Series with cluster numbers.
        """

        if dbscan:
            # Apply DBSCAN clustering
            name = 'DBSCAN'
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(df)

        else:
            # Apply K-Means clustering
            name = 'K-MEANS'
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(df)

        # Generate the plot if requested
        if plot:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='viridis')
            plt.title(f'Clustering with {name}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()

        # Create a series with cluster numbers
        cluster_series = pd.Series(labels, name='Clusters', index=df.index)

        return cluster_series

    @classmethod
    def generate_soft_clusters(cls, df: pd.DataFrame, num_clusters=5, plot=False, random_state=np.random.randint(1, 1000), n_init=10):
        """
        Applies Gaussian Mixture Models (GMM) to a DataFrame and returns the probabilities of belonging
        to each of the clusters.

        :param df: DataFrame with the data for analysis.
        :param num_clusters: Number of clusters to generate.
        :param plot: If True, generates a plot of the clusters.
        :param random_state (opt): Seed to use, random by default.
        :param n_init (opt): number of times the Expectation-Maximization (EM) algorithm will
            attempt to converge to an optimal result, 10 by default.

        :return: DataFrame with probabilities of belonging to each cluster.

        Note: Data should be scaled for good performance.
        """

        # Apply Gaussian Mixture Models
        gmm = GaussianMixture(n_components=num_clusters, random_state=random_state, n_init=n_init)
        gmm.fit(df)
        probabilities = gmm.predict_proba(df)

        # Generate the plot if requested
        if plot:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=probabilities, cmap='viridis')
            plt.title('Soft Clustering with Gaussian Mixture Models')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.colorbar()
            plt.show()

        # Create a DataFrame with the probabilities of belonging to each cluster
        cluster_probabilities = pd.DataFrame(probabilities, columns=[f'Cluster_{i}' for i in range(num_clusters)], index=df.index)

        return cluster_probabilities
    

class Graphs_ml:

    @classmethod
    def plot_classification(cls, y_true, y_pred)->None:
        """
        Generates an interactive confusion matrix to evaluate the model's performance.

        Parameters:
            y_true (array-like): The true values of the target variable.
            y_pred (array-like): The values predicted by the model.

        Returns:
            None
        """
        # Check if the values are numeric or categorical
        if not np.issubdtype(y_true.dtype, np.number):
            y_true = pd.Categorical(y_true)
        if not np.issubdtype(y_pred.dtype, np.number):
            y_pred = pd.Categorical(y_pred)

        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Generate the interactive confusion matrix using Plotly
        fig = go.Figure(data=go.Heatmap(z=cm, x=['Class 0', 'Class 1'], y=['Class 0', 'Class 1'],
                                       colorscale='YlGnBu', zmin=0, zmax=cm.max().max()))

        # Show the values within the chart
        for i in range(len(cm)):
            for j in range(len(cm)):
                fig.add_annotation(
                    x=['Class 0', 'Class 1'][j],
                    y=['Class 0', 'Class 1'][i],
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(size=14, color='white' if cm[i, j] > cm.max().max() / 2 else 'black')
                )

        fig.update_layout(title='Confusion Matrix',
                          xaxis_title='Predicted Values',
                          yaxis_title='True Values',
                          xaxis=dict(side='top'))

        fig.show()

    @classmethod
    def plot_regression(cls, y_true, y_pred)->None:
        """
        Generates an interactive scatter plot to compare true values and predicted values in a regression problem.
    
        Parameters:
            y_true (array-like): The true values of the target variable.
            y_pred (array-like): The values predicted by the model.
    
        Returns:
            None
        """
        # Check if the values are numeric
        if not np.issubdtype(y_true.dtype, np.number):
            raise ValueError("This method is only valid for regression problems with numeric variables.")
    
        # Create a DataFrame for true and predicted values
        df = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
    
        # Generate the interactive scatter plot using Plotly
        fig = px.scatter(df, x='True', y='Predicted', labels={'True': 'True Values', 'Predicted': 'Predicted Values'},
                         title='Comparison Between True Values and Predicted Values (Regression)')
        fig.add_trace(go.Scatter(x=[min(y_true), max(y_true)], y=[min(y_true), max(y_true)], mode='lines', line=dict(color='red'),
                                 name='45-Degree Line'))
    
        fig.show()

    @classmethod
    def plot_cluster(cls, df, random_state=np.random.randint(1, 1000)):
        """
        Performs clustering analysis using the Elbow Method and Silhouette Score.

        :param df: DataFrame containing the data for analysis.
        :param random_state (optional): Seed value to use.

        Example of use:
            Graphs.plot_cluster(df). The df should be clean (without missing values or categorical variables).

        Note: Use scaled data for better results.
        """

        # Elbow Method
        def elbow_method(data, max_clusters):
            distortions = []
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
                kmeans.fit(data)
                distortions.append(kmeans.inertia_)
            return distortions

        max_clusters = 10
        distortions = elbow_method(df, max_clusters)

        # Finding the optimal number of clusters using silhouette score
        silhouette_scores = []
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            labels = kmeans.fit_predict(df)
            silhouette_avg = silhouette_score(df, labels)
            silhouette_scores.append(silhouette_avg)

        # Create a figure with two subplots vertically
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

        # Plot the elbow method on the first subplot
        ax1.plot(range(1, max_clusters + 1), distortions, marker='o')
        ax1.set_title('Elbow Method')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Distortion')

        # Plot silhouette scores on the second subplot
        ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        ax2.set_title('Silhouette Score')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')

        # Adjust spaces between subplots
        plt.tight_layout()
        plt.show()


class ML(Tools):

    @classmethod
    def lightgbm_model(cls, df:pd.DataFrame, target:str, problem_type:str, random_state=np.random.randint(1,1000),
                        num_leaves=20, num_boost_round=100, graph=False, test_size=0.2, cv=5,
                        learning_rate=0.1, max_depth=-1, save_model=False, model_filename='lightgbm', 
                        encode_categorical=False, grid='', boosting_type='gbdt', n_iter=10):
        """
        Use LightGBM to predict the target variable in a DataFrame.
        
        Parameters:
            df (pandas DataFrame): The DataFrame containing the input variables and the target variable.
            target (str): The name of the column containing the target variable.
            problem_type (str): Type of problem: 'classification' or 'regression'.
            random_state (int): Seed to use for data splitting, defaults to a random number.
            num_leaves (int): Maximum number of leaves in each tree. Controls the model's complexity. Default is 20.
            num_boost_round (int): The number of algorithm iterations (number of trees), default is 100.
            graph (bool): If True, generate corresponding graphs based on the problem type.
            test_size (float): The sample size for the test set, default is 0.2.
            cv: Number of cross-validation partitions. Default is 5.
            n_iter: Number of hyperparameter combinations to try. Default is 10.
            learning_rate (float): Model's learning rate, default is 0.1.
            max_depth (int): Maximum tree depth, default is -1 (no limit).
            save_model (bool): If True, the trained model will be saved to disk. Default is False.
            model_filename (str): The filename for saving the model. Required if save_model is True.
            encode_categorical (bool, optional):
            grid (str, optional): Indicates whether to perform hyperparameter tuning using: 
                'full', 'random' or 'optuna'. Default None.
            boosting_type (str): Type of boosting algorithm to use.
                Available options:
                - 'gbdt': Gradient Boosting Decision Tree (default).
                - 'dart': Dropouts meet Multiple Additive Regression Trees.
                - 'goss': Gradient-based One-Side Sampling.
            NOTE: To load a model, do the following:
                import joblib
        
                # Path and filename where the model was saved
                model_filename = "model_file_name.pkl"
        
                # Load the model
                loaded_model = joblib.load(model_filename)
        
                # You can now use the loaded model to make predictions
                # Suppose you have a dataset 'X_test' for predictions
                y_pred = loaded_model.predict(X_test)
        
        Returns:
            print(pd.DataFrame): A DataFrame containing various metrics and statistics of the model.
            LightGBM Model: The trained model.
        """

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = cls.split_and_convert_data(df, target,
                                                                           test_size=test_size,
                                                                           random_state=random_state,
                                                                           encode_categorical=encode_categorical)

        if problem_type == 'classification' and y_train.nunique() > 2:
                raise NotImplementedError('Sorry, multiclass classification not found!')

        if grid in ['full', 'random']:
            # Define the hyperparameter search space
            params = {
                'boosting_type': ['gbdt', 'dart', 'goss'],
                'num_boost_round': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [2, 3, 5],
            }
        
            # Define the estimator and scoring based on the problem type
            estimator = lgb.LGBMClassifier() if problem_type=='classification' else lgb.LGBMRegressor()
            scoring = 'neg_log_loss' if problem_type=='classification' else 'neg_mean_squared_error'

            # Perform grid search or random search using the Tools method
            if grid == 'random':
                best_params = cls._random_search(estimator, 
                                                params, 
                                                X_train, y_train,
                                                scoring, 
                                                cv=cv,
                                                n_iter=n_iter)
            else:
                best_params = cls._grid_search(estimator, 
                                                params, 
                                                X_train, y_train,
                                                scoring, 
                                                cv=cv,
                                                n_iter=n_iter)
            # Display the values selected by Search
            print(pd.DataFrame(best_params))
            # Use the best-found hyperparameters
            params = best_params['best_params']

        elif grid == 'optuna':
            params = {
                'max_depth': (1, 10),
                'learning_rate': (0.001, 1.0),
                'n_estimators': (50, 1000),
                'subsample': (0.1, 1.0),
            }

            # Define the estimator and scoring based on the problem type
            estimator = lgb.LGBMClassifier() if problem_type=='classification' else lgb.LGBMRegressor()
            scoring = 'neg_log_loss' if problem_type=='classification' else 'neg_mean_squared_error'

            best_params = cls._optuna_search(estimator, 
                                            params, 
                                            X_train, y_train,
                                            scoring, 
                                            cv=cv,
                                            n_iter=n_iter)
            # Display the values selected by Optuna
            print(pd.DataFrame(best_params))
            # Use the best-found hyperparameters
            params = best_params['best_params']


        else:
            # Manual parameters for the LightGBM model
            params = {
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'boosting_type': boosting_type,
                'num_boost_round': num_boost_round
            }

        if problem_type == 'classification':
            # Binary classification problem
            params['objective'] = ['binary'] if y_train.nunique() == 2 else ['multiclass']
            params['metric'] = ['binary_logloss'] if y_train.nunique() == 2 else ['multi_logloss']
            # Create the LightGBM model with the best hyperparameters and train it
            lgb_model = lgb.LGBMClassifier(**params)
            metric_type='classification'

        elif problem_type == 'regression':
            # Regression problem
            params['objective'] = 'regression'
            params['metric'] = 'l2'  # Mean Squared Error (MSE)
            # Create the LightGBM model with the best hyperparameters and train it
            lgb_model = lgb.LGBMRegressor(**params)
            metric_type='regression'

        else:
            raise ValueError("El parámetro 'problem_type' debe ser 'classification' o 'regression'.")

        # Train the model
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=50) # type: ignore
        
        # Make predictions on the test set
        y_pred = lgb_model.predict(X_test)
        
        # Calculate classification metrics
        metrics = cls._metrics(y_test, y_pred, metric_type=metric_type)
        
        if graph and problem_type == 'classification':
            Graphs_ml.plot_classification(y_test, y_pred)
        elif graph and problem_type == 'regression':
            Graphs_ml.plot_regression(y_test, y_pred)

        if save_model and model_filename:
            # Save the trained model to disk
            joblib.dump(lgb_model, f'{model_filename}.pkl')

        print(metrics)
        return lgb_model

    @classmethod
    def xgboost_model(cls, df:pd.DataFrame, target:str, problem_type:str, test_size=0.2, cv=5,
                n_estimators=100, save_model=False, model_filename='xgboost',
                learning_rate=0.1, max_depth=3, random_state=np.random.randint(1, 1000),
                graph=False, grid='', n_iter=10):
        """
        Use XGBoost to predict the target variable in a DataFrame.

        Parameters:
            df (pandas DataFrame): The DataFrame containing the input variables and the target variable.
            target (str): The name of the column containing the target variable.
            problem_type (str): Type of problem: 'classification' or 'regression'.
            test_size (float): The sample size for the test set, default is 0.2.
            n_estimators (int): The number of algorithm iterations (number of trees), default is 100.
            learning_rate (float): Model's learning rate, default is 0.1.
            max_depth (int): Maximum tree depth, default is 3.
            cv: Number of cross-validation partitions. Default is 5.
            n_iter: Number of hyperparameter combinations to try. Default is 10.
            random_state (int): Seed to use for data splitting, defaults to a random number.
            graph (bool): If True, generate corresponding graphs based on the problem type.
            grid (str, optional): Indicates whether to perform hyperparameter tuning using: 
                'full', 'random' or 'optuna'. Default None.
            NOTE: To load a model, do the following:
                import joblib

                # Path and filename where the model was saved
                model_filename = "model_file_name.pkl"

                # Load the model
                loaded_model = joblib.load(model_filename)

                # You can now use the loaded model to make predictions
                # Suppose you have a dataset 'X_test' for predictions
                y_pred = loaded_model.predict(X_test)

        Returns:
            print(pd.DataFrame): A DataFrame containing various metrics and statistics of the model.
            XGBoost Model: The trained model.
        """


        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = cls.split_and_convert_data(df,
                                                                      target,
                                                                      test_size=test_size,
                                                                      random_state=random_state)

        if grid in ['full', 'random']:
            # XGBoost model parameters
            params = {
                'booster': ['gbtree', 'gblinear', 'dart'],
                'n_estimators': [50, 100, 200],
                'max_depth': [1, 2, 3, 5],  # Maximum tree depth
                'learning_rate': [0.1, 0.06, 0.03, 0.01],  # Learning rate
                'subsample': [0.1, 0.2, 0.5, 1.0]  # Subsampling rate
            }
            
            # Define the estimator and scoring based on the problem type
            estimator = xgb.XGBClassifier() if problem_type=='classification' else xgb.XGBRegressor()
            scoring = 'neg_log_loss' if problem_type=='classification' else 'neg_mean_squared_error'
            # Perform grid search using the Tools method

            if grid == 'random': 
                best_params = cls._random_search(estimator, 
                                                params, 
                                                X_train, y_train,
                                                scoring, 
                                                cv=cv,
                                                n_iter=n_iter)
            else:
                best_params = cls._grid_search(estimator, 
                                                params, 
                                                X_train, y_train,
                                                scoring, 
                                                cv=cv,
                                                n_iter=n_iter)
                
            # Display the values selected by GridSearch
            print(pd.DataFrame(best_params))
            # Use the best-found hyperparameters
            params = best_params['best_params']

        elif grid == 'optuna':
            params = {
                'n_estimators': (50, 200),
                'max_depth': (1, 5),
                'learning_rate': (0.001, 0.1),
                'subsample': (0.1, 1.0)
            }

            # Define the estimator and scoring based on the problem type
            estimator = xgb.XGBClassifier() if problem_type=='classification' else xgb.XGBRegressor()
            scoring = 'neg_log_loss' if problem_type=='classification' else 'neg_mean_squared_error'
            
            # Perform grid search using the Tools method
            best_params = cls._optuna_search(estimator, 
                                            params, 
                                            X_train, y_train,
                                            scoring, 
                                            cv=cv,
                                            n_iter=n_iter)
            
            # Display the values selected by Optuna
            print(pd.DataFrame(best_params))
            # Use the best-found hyperparameters
            params = best_params['best_params']

        else:
            params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators
            }

        if problem_type == 'classification':
            # Binary or multiclass classification problem
            params['objective'] = 'binary:logistic' if y_test.nunique() == 2 else 'multi:softmax'
            
            # Train the XGBoost model for classification
            xgb_model = xgb.XGBClassifier(**params, num_class=y_test.nunique())
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
            
            # # Make predictions on the test set
            y_pred = xgb_model.predict(X_test)
            if params['objective'] == 'binary:logistic':
                y_pred_binary = np.where(y_pred > 0.5, 1, 0)
            else:
                y_pred_binary = y_pred #np.argmax(y_pred, axis=1)
            metrics = cls._metrics(y_test, y_pred_binary, metric_type='classification')
            if graph == True:
                Graphs_ml.plot_classification(y_test, y_pred_binary)

        elif problem_type == 'regression':
            # Regression problem
            params['objective'] = 'reg:squarederror'

            # Train the XGBoost model for regression
            xgb_model = xgb.XGBRegressor(**params)
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
            
            # Make predictions on the test set
            y_pred = xgb_model.predict(X_test)
            metrics = cls._metrics(y_test, y_pred, metric_type='regression')

        else:
            raise ValueError("The 'problem_type' parameter must be 'classification' or 'regression'.")
        
        if save_model and model_filename:
            # Save the trained model to disk
            joblib.dump(xgb_model, f'{model_filename}.pkl')
        
        print(metrics)
        return xgb_model

    @classmethod
    def catboost_model(cls, df:pd.DataFrame, target:str, problem_type:str, test_size=0.2, n_iter=10,
                        num_boost_round=100, learning_rate=0.1, max_depth=3, cv=5,
                        random_state=np.random.randint(1, 1000), graph=False,
                        save_model=False, model_filename='catboost', grid=False):
        """
        Use CatBoost to predict the target variable in a DataFrame.

        Parameters:
            df (pandas DataFrame): The DataFrame containing the input variables and the target variable.
            target (str): The name of the column containing the target variable.
            problem_type (str): Type of problem: 'classification' or 'regression'.
            test_size (float): The sample size for the test set, default is 0.2.
            num_boost_round (int): The number of algorithm iterations (number of trees), default is 100.
            learning_rate (float): Model's learning rate, default is 0.1.
            max_depth (int): Maximum tree depth, default is 3.
            cv: Number of cross-validation partitions. Default is 5.
            n_iter: Number of hyperparameter combinations to try. Default is 10.
            random_state (int): Seed to use for data splitting, defaults to a random number.
            graph (bool): If True, generate corresponding graphs based on the problem type.
            save_model (bool): If True, the trained model will be saved to disk. Default is False.
            model_filename (str): The filename for saving the model if save_model is True.
            grid (str, optional): Indicates whether to perform hyperparameter tuning using: 
                'full', 'random' or 'optuna'. Default None.
            NOTE: To load a model, do the following:
                import joblib

                # Path and filename where the model was saved
                model_filename = "model_file_name.pkl"

                # Load the model
                loaded_model = joblib.load(model_filename)

                # You can now use the loaded model to make predictions
                # Suppose you have a dataset 'X_test' for predictions
                y_pred = loaded_model.predict(X_test)

        Returns:
            print(pd.DataFrame): A DataFrame containing various metrics and statistics of the model.
            CatBoost Model: The trained model.
        """


        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = cls.split_and_convert_data(df,
                                                                      target,
                                                                      test_size=test_size,
                                                                      random_state=random_state)
        
        if problem_type == 'classification' and y_train.nunique() > 2:
                raise NotImplementedError('Sorry, multiclass classification not found!')

        if grid in ['full', 'random']:
            params = {
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'depth': [2, 3, 5],
                'num_boost_round': [50, 100, 200]
            }

            # Define the estimator and scoring based on the problem type
            estimator = cb.CatBoostClassifier() if problem_type=='classification' else cb.CatBoostRegressor()
            scoring = 'neg_log_loss' if problem_type=='classification' else 'neg_mean_squared_error'

            # Perform grid search using the Tools method
            if grid == 'random':
                best_params = cls._random_search(estimator, 
                                                params, 
                                                X_train, y_train,
                                                scoring, 
                                                cv=cv,
                                                n_iter=n_iter)
            else:
                best_params = cls._grid_search(estimator, 
                                                params, 
                                                X_train, y_train,
                                                scoring, 
                                                cv=cv,
                                                n_iter=n_iter)
                
            # Display the values selected by GridSearch
            print(pd.DataFrame(best_params))
            # Use the best-found hyperparameters
            params = best_params['best_params']

        elif grid == 'optuna':
            params = {
                'learning_rate': (0.001, 0.1),
                'depth': (2, 5),
                'num_boost_round': (50, 200)
            }
            # Define the estimator and scoring based on the problem type
            estimator = cb.CatBoostClassifier() if problem_type=='classification' else cb.CatBoostRegressor()
            scoring = 'neg_log_loss' if problem_type=='classification' else 'neg_mean_squared_error'

            best_params = cls._optuna_search(estimator, 
                                            params, 
                                            X_train, y_train,
                                            scoring, 
                                            cv=cv,
                                            n_iter=n_iter)
            # Display the values selected by Optuna
            print(pd.DataFrame(best_params))
            # Use the best-found hyperparameters
            params = best_params['best_params']

        else:
            params = {
                'num_boost_round': num_boost_round,
                'learning_rate': learning_rate,
                'depth': max_depth,
                'random_state': random_state
            }

        # Create the CatBoost model
        if problem_type == 'classification':
            params['loss_function'] = 'Logloss'
            model = cb.CatBoostClassifier(**params)
        elif problem_type == 'regression':
            params['loss_function'] = 'RMSE'
            model = cb.CatBoostRegressor(**params)
        else:
            raise ValueError("The 'problem_type' parameter must be 'classification' or 'regression'.")

        # Train the CatBoost model
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10, verbose=50)
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        # Calculate classification or regression metrics
        if problem_type == 'classification':
            metrics = cls._metrics(y_test, y_pred, metric_type='classification')
        else:
            metrics = cls._metrics(y_test, y_pred, metric_type='regression')

        if graph:
            if problem_type == 'classification':
                Graphs_ml.plot_classification(y_test, y_pred)
            else:
                Graphs_ml.plot_regression(y_test, y_pred)

        if save_model and model_filename:
            # Save the trained model to disk
            joblib.dump(model, f'{model_filename}.pkl')

        print(metrics)

        return model
