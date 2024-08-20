"""
Author: ale-uy
Date: 05/2023
Updated: 08/2024
Version: v2.1
File: eda.py
Description: Automate data analysis and cleaning processes
License: Apache License Version 2.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTENC
from typing import Literal, Optional, Union



class EDA:

    @classmethod
    def analyze_nulls(cls, df: pd.DataFrame) -> None:
        """
        Returns the percentage of null values in the entire dataset for each column.
        Returns:
            Series: A series containing the percentages of null values for each column,
                    sorted from highest to lowest.
        """
        print(df.isna().sum().sort_values(ascending=False) / df.shape[0] * 100)

    @classmethod
    def numeric_statistics(cls, df: pd.DataFrame) -> None:
        """
        Generates statistical data for numeric variables in the DataFrame.
        Returns:
            DataFrame: A DataFrame containing statistics for numeric variables,
                       including count, mean, std, min, 25%, 50%, 75%, and max.
        """
        print(df.select_dtypes('number').describe().T)

    @classmethod
    def convert_to_numeric(cls, df: pd.DataFrame, 
                           method: Literal["dummy", "ohe", "label"] = "ohe", 
                           drop_first: bool = True) -> pd.DataFrame:
        """
        Performs the encoding of categorical variables using different methods.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing the variables to be encoded.
            method (Literal["dummy", "ohe", "label"]): The encoding method to use. Default is "ohe".
            drop_first (bool): Drops the first dummy when using "dummy" or "ohe" encoding. Default is True.
        
        Returns:
            pd.DataFrame: The original DataFrame with the categorical columns encoded, excluding the target column.
        
        Raises:
            ValueError: If an invalid method is provided.
        """
        
        encoded_df = df.copy()
        
        object_columns = encoded_df.select_dtypes(include=['object']).columns
        
        if method == "dummy":
            encoded_df = pd.get_dummies(encoded_df, columns=object_columns, drop_first=drop_first)
        
        elif method == "ohe":
            encoder = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)
            encoded_cats = encoder.fit_transform(encoded_df[object_columns])
            encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(object_columns))
            encoded_df = pd.concat([encoded_df.drop(columns=object_columns), encoded_cats_df], axis=1)
        
        elif method == "label":
            for col in object_columns:
                encoded_df[col] = LabelEncoder().fit_transform(encoded_df[col])
        
        else:
            raise ValueError("The 'method' parameter must be one of: 'dummy', 'ohe', 'label'.")
        
        return encoded_df

    @classmethod
    def remove_missing_if(cls, df:pd.DataFrame, p=0.5):
        """
        Deletes columns that have a percentage greater than or equal to p of null values.
        Parameters:
            p (float): The threshold percentage to consider a column with null values.
                       Default is 0.5 (50%).
        """

        nan_percentages = df.isna().mean()
        columns_to_drop = nan_percentages[nan_percentages >= p].index
        aux = df.drop(columns=columns_to_drop)
        return aux

    @classmethod
    def impute_missing(cls, df: pd.DataFrame, 
                       method: Literal['mm', 'knn'] = 'mm', 
                       num_strategy: Literal['mean', 'median'] = 'median',
                       n_neighbors: Optional[int] = None) -> pd.DataFrame:
        """
        Imputes missing values in a DataFrame using different methods.

        Parameters:
            df (pd.DataFrame): The DataFrame containing missing values.
            method (Literal['mm', 'knn']): The imputation method to use. Default is 'mm'.
                'mm' (Median/Mode): Imputes median for numeric and mode for categorical variables.
                'knn' (K-Nearest Neighbors): Imputes missing values using KNNImputer.
                'interpolate': Imputes missing values using interpolation.
            num_strategy (Literal['mean', 'median']): The strategy to use for imputing numeric columns when method='mm'.
                                                      Default is 'median'.
            n_neighbors (Optional[int]): The number of nearest neighbors to use in the KNNImputer method.
                                         Only applicable if the method is 'knn'. If None, performs a search.

        Returns:
            pd.DataFrame: The original DataFrame with missing values imputed.

        Raises:
            ValueError: If an invalid method is provided.

        Usage Example:
            imputed_df = EDA.impute_missing(df, method="knn", n_neighbors=5)
        """
        df_imputed = df.copy()

        if method == 'knn':
            if n_neighbors is None:
                param_grid = {'n_neighbors': list(range(3, 16))}
                knn_imputer = KNNImputer()
                grid_search = RandomizedSearchCV(knn_imputer, param_grid, cv=3, n_iter=6, n_jobs=-1)
                grid_search.fit(df_imputed)
                n_neighbors = grid_search.best_params_['n_neighbors']
                print(f"Best value of n_neighbors found: {n_neighbors}")

            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed_values = knn_imputer.fit_transform(df_imputed)
            df_imputed = pd.DataFrame(imputed_values, columns=df.columns, index=df.index)

        elif method == 'mm':
            for col in df_imputed.columns:
                if num_strategy == 'median':
                        df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                elif num_strategy == 'mean':
                    df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
                else:
                    df_imputed[col].fillna(df_imputed[col].mode().iloc[0], inplace=True)

        elif method == 'interpolate':
            df_imputed.interpolate(method='linear', inplace=True)

        else:
            raise ValueError("The 'method' parameter must be one of: 'mm', 'knn' or 'interpolate'.")

        return df_imputed
        
    @classmethod
    def remove_single_value_columns(cls, df:pd.DataFrame):
        """
        Remove variables that have only one unique value
        """
        columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
        df_copy = df.drop(columns=columns_to_drop)
        return df_copy
    
    @classmethod
    def shuffle_data(cls, df:pd.DataFrame):
        """
        Randomly shuffles the data in the DataFrame.
        Parameters:
            df (pandas DataFrame): The DataFrame with the data to shuffle.
        Returns:
            pandas DataFrame: A new DataFrame with rows randomly shuffled.
        """
        shuffled_df = df.sample(frac=1, random_state=np.random.randint(1, 1000))
        return shuffled_df
    
    @classmethod
    def standardize_variables(cls, df: pd.DataFrame, method: Literal["zscore", "minmax", "robust"] = "zscore") -> pd.DataFrame:
        """
        Standardizes numeric variables in a DataFrame using the specified method.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the variables to standardize.
            method (Literal["zscore", "minmax", "robust"]): The standardization method to use.
                - 'zscore': Standardization using Z-Score (mean 0, standard deviation 1).
                - 'minmax': Standardization using Min-Max (range 0 to 1).
                - 'robust': Robust standardization using medians and quartiles.
                Default is "zscore".

        Returns:
            pd.DataFrame: A new DataFrame with standardized numeric variables.

        Raises:
            ValueError: If an invalid method is provided or if there are no numeric columns.
        """
        numeric_columns = df.select_dtypes(include='number').columns

        if numeric_columns.empty:
            raise ValueError("No numeric columns found in the DataFrame.")

        scaler = {
            'zscore': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }.get(method)

        if scaler is None:
            raise ValueError("The 'method' parameter must be one of: 'zscore', 'minmax', 'robust'.")

        result_df = df.copy()
        result_df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        return result_df
    
    @classmethod
    def apply_log1p_transformation(cls, df: pd.DataFrame):
        """
        Apply np.log1p transformation to non-negative numerical columns in a DataFrame.

        Parameters:
        - df (DataFrame): The pandas DataFrame containing the data.

        This method applies the np.log1p transformation (logarithm of 1 plus the value) to all non-negative numerical columns
        in the DataFrame for improved numerical stability.

        Args:
        - df (DataFrame): The pandas DataFrame containing the data.

        Returns:
        - df (DataFrame): The DataFrame with transformed columns.
        """
        df_transformed = df.copy()
        numeric_columns = df.select_dtypes(include=['float', 'int']).columns
        for column in numeric_columns:
            if (df_transformed[column] >= 0).all():
                df_transformed[column] = np.log1p(df_transformed[column])

        return df_transformed

    @classmethod
    def balance_data(cls, df: pd.DataFrame, target: str, method: str = "oversampling", random_state: Union[int, None] = None) -> pd.DataFrame:
        """
        Balances an imbalanced dataset in a binary classification task using oversampling or undersampling.

        Parameters:
            df (pd.DataFrame): The DataFrame containing input variables and the target variable.
            target (str): The name of the column containing the target variable.
            method (str): The balancing method to use. Options:
                - "oversampling": Uses the SMOTENC algorithm for oversampling (default).
                - "undersampling": Uses random undersampling of the majority class.
            random_state (int or None): Seed for random number generation. Default is None.

        Returns:
            pd.DataFrame: A new DataFrame with balanced classes.

        Raises:
            ValueError: If an invalid method is provided or if the target is not binary.

        Example:
            # Balance classes using oversampling
            df_balanced = EDA.balance_data(df, 'target_variable', method='oversampling')
        """
        if method not in ["oversampling", "undersampling"]:
            raise ValueError("The 'method' parameter must be either 'oversampling' or 'undersampling'.")

        # Ensure target is binary
        if df[target].nunique() != 2:
            raise ValueError("The target variable must be binary for this balancing method.")

        X = df.drop(columns=[target])
        y = df[target]

        if method == "oversampling":
            categorical_cols = X.select_dtypes(exclude=['number']).columns
            categorical_features = [X.columns.get_loc(col) for col in categorical_cols]

            smote_nc = SMOTENC(categorical_features=categorical_features, random_state=random_state)
            X_resampled, y_resampled = smote_nc.fit_resample(X, y)

            df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                                     pd.Series(y_resampled, name=target)], axis=1)

        else:  # undersampling
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            minority_class = y_encoded.min()
            majority_class = y_encoded.max()
            
            df_minority = df[y_encoded == minority_class]
            df_majority = df[y_encoded == majority_class]
            
            df_majority_reduced = df_majority.sample(df_minority.shape[0], random_state=random_state)
            df_balanced = pd.concat([df_minority, df_majority_reduced], axis=0)
            df_balanced = df_balanced.sample(frac=1, random_state=random_state)

        return df_balanced
        
    @classmethod
    def remove_duplicate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame from which duplicate rows will be removed.
            
        Returns:
            pd.DataFrame: A new DataFrame with duplicate rows removed.
        """
        print(f'Number of duplicate observations: {df.duplicated().sum()}')
        print(f'Percentage of duplicate observations in total observations: % \
              {100*(df.duplicated().sum())/df.shape[0]:.1f}')
        cleaned_df = df.drop_duplicates(keep='first')
        return cleaned_df
        
    @classmethod
    def remove_outliers(cls, df: pd.DataFrame, method='zscore', threshold=np.inf) -> pd.DataFrame:
        """
       Remove outliers from a pandas DataFrame using different methods.

       Arguments:
           df (pd.DataFrame): The data frame from which outliers will be removed.
           method (str): The method to remove outliers, can be 'zscore' (default) or 'iqr'.
           threshold (float): The threshold for considering a value as an outlier.

       Returns:
           pd.DataFrame: A new DataFrame with no outliers.
       """
        if method == 'zscore':
            z_scores = np.abs((df - df.mean()) / df.std())
            cleaned_df = df[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            cleaned_df = df[((df >= lower_bound) & (df <= upper_bound)).all(axis=1)]
        else:
            raise ValueError("Invalid method. Please use 'zscore' or 'iqr'.")

        return cleaned_df

    @classmethod
    def perform_pca(cls, df, n_components='mle'):
        """
        Perform Principal Component Analysis (PCA).

        Parameters:
        - df (pd.DataFrame or array-like): The data to perform PCA on.
        - n_components (str or int): Number of components to keep. You can use 'mle' for automatic selection
          or specify an integer for a fixed number of components.

        Returns:
        - transformed_df (pd.DataFrame): The data transformed to the new feature space.
        """
        pca = PCA(n_components=n_components)
        transformed_df = pca.fit_transform(df)
        return pd.DataFrame(transformed_df, columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])


class Graphs_eda:

    @classmethod
    def categorical_plots(cls, df: pd.DataFrame) -> None:
        """
        Creates horizontal bar charts for each categorical variable in the DataFrame.
        Parameters:
            df (pandas DataFrame): The DataFrame containing the categorical variables to plot.
        """
        # Select categorical columns from the DataFrame
        categorical_columns = df.select_dtypes('O').columns
        # Calculate the number of categorical columns and rows to organize the plots
        num_columns = len(categorical_columns)
        rows = (num_columns + 1) // 2
        # Create the figure and axes for the plots
        _, ax = plt.subplots(nrows=rows, ncols=2, figsize=(16, rows * 6))
        ax = ax.flat
        # Generate horizontal bar charts for each categorical variable
        for i, col in enumerate(categorical_columns):
            df[col].value_counts().plot.barh(ax=ax[i])
            ax[i].set_title(col, fontsize=12, fontweight="bold")
            ax[i].tick_params(labelsize=12)
        # Adjust the layout and display the plots
        plt.tight_layout()
        plt.show()

    @classmethod
    def numerical_plot_density(cls, df: pd.DataFrame) -> None:
        """
        Generate density plots for all numerical features in a DataFrame.

        Parameters:
        - df (DataFrame): The pandas DataFrame containing the data.

        This method iterates through all the numerical columns in the DataFrame, creating density plots for each one.
        The density plots display the probability density of each numerical feature.

        The plots are organized in a grid, with two plots per row, and the styling is set to 'whitegrid'.

        Args:
        - df (DataFrame): The pandas DataFrame containing the data.

        Returns:
        None
        """
        # Get the list of numerical columns in your DataFrame
        numeric_columns = df.select_dtypes(include=['float', 'int']).columns

        # Set the Seaborn style
        sns.set(style="whitegrid")

        # Define the plot size and the number of rows and columns in the grid
        num_plots = len(numeric_columns)
        rows = (num_plots + 1) // 2  # Calculate the number of rows needed (two plots per row)
        cols = 2  # Two plots per row
        _, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

        # Iterate through the numerical features and create the density plots
        for i, feature_name in enumerate(numeric_columns):
            row_idx, col_idx = divmod(i, cols)  # Calculate the current row and column index
            sns.histplot(data=df, x=feature_name, kde=True, ax=axes[row_idx, col_idx])
            axes[row_idx, col_idx].set_title(f'Density Plot of {feature_name}')
            axes[row_idx, col_idx].set_xlabel('Feature Value')
            axes[row_idx, col_idx].set_ylabel('Density')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plots
        plt.show()

    @classmethod
    def histogram_plot(cls, df: pd.DataFrame, column: str) -> None:
        """
        Generates an interactive histogram for a specific column in the DataFrame.
        Parameters:
            column (str): Name of the column to visualize in the histogram.
        """
        fig = px.histogram(df, x=column)
        fig.show()

    @classmethod
    def box_plot(cls, df: Union[pd.Series, pd.DataFrame]) -> None:
        """
        Generates a combined boxplot for numeric variables in a DataFrame or a single Series.

        Args:
        df (Union[pd.Series, pd.DataFrame]): The pandas DataFrame or Series containing the data.

        Raises:
        ValueError: If the input is not a pandas DataFrame or Series, or if it does not contain numeric variables.

        Returns:
        None
        """
        if isinstance(df, pd.Series):
            data = pd.DataFrame(df)
        elif isinstance(df, pd.DataFrame):
            data = df.copy()
        else:
            raise ValueError("The input must be a pandas DataFrame or Series.")

        numeric_data = data.select_dtypes(include=['float', 'int'])
        
        if numeric_data.empty:
            raise ValueError("The DataFrame or Series does not contain numeric variables.")

        df_melted = pd.melt(numeric_data)
        
        custom_colors = px.colors.qualitative.Plotly

        fig = px.box(df_melted, x='variable', y='value', color='variable', 
                    color_discrete_sequence=custom_colors, 
                    title='Box Plot')
        
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Values",
            showlegend=False
        )
        
        fig.show()

    @classmethod
    def hierarchical_clusters_plot(cls, df: pd.DataFrame, method='single', metric='euclidean', save_clusters=False):
        """
        Generates a hierarchical dendrogram from a DataFrame and saves the clusters in a series.

        :param df: DataFrame with data for analysis.
        :param method: 'single' (default), 'complete', 'average', 'weighted', 'ward', 'centroid', 'median'.
        :param metric: Distance metric for linkage calculation.
        :param save_clusters: True to generate a series with the cluster of each observation, \
            will prompt for the number of clusters to use.

        :return: Series with generated clusters.
        """
        # Calculate the linkage matrix
        linked = linkage(df.values, method=method, metric=metric, optimal_ordering=True)

        # Generate the dendrogram
        plt.figure(figsize=(10, 6))
        dendrogram(linked, orientation='top', labels=df.index, distance_sort='descending', show_leaf_counts=True)
        plt.title('Dendrogram')
        plt.xlabel('Indices of Points')
        plt.ylabel('Distances')
        plt.show()

        if save_clusters:
            num_clusters = int(input('Number of clusters to use (observe the dendrogram)'))
            # Get clusters using the fcluster function
            corte = linked[-num_clusters + 1, 2]  # Height of the cut in the dendrogram
            clusters = fcluster(linked, corte, criterion='distance')

            # Create a series with the clusters and return it
            cluster_series = pd.Series(clusters, index=df.index, name='Clusters')
            return cluster_series

    @classmethod
    def correlation_heatmap_plot(cls, df: pd.DataFrame, method='spearman') -> None:
        """
        Generates a correlation heatmap for the given DataFrame.

        Args:
            df (pandas DataFrame): A DataFrame with the data.
            method (str, optional): 'pearson', 'kendall' or 'spearman' (default)

        Returns:
            None.
        """
        corr = df.corr(method=method)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, linewidth=0.5, annot=True, cmap="RdBu", vmin=-1, vmax=1)

    @classmethod
    def pca_elbow_method_plot(cls, df: pd.DataFrame, target_variance=0.95) -> None:
        """
        Perform PCA and use the elbow method to select the number of components.

        Parameters:
        - df (pd.DataFrame or array-like): The data to perform PCA on.
        - target_variance (float): The target cumulative explained variance.

        Print:
        - num_components (int): The number of components selected.
        """
        pca = PCA()
        pca.fit(df)

        explained_variance = pca.explained_variance_ratio_
        cum_variance = np.cumsum(explained_variance)

        # Calculate x values with a step of 2
        x_values = list(range(1, len(explained_variance) + 1, 1))
        cum_variance_values = [cum_variance[i - 1] for i in x_values]

        fig = px.line(x=x_values, y=cum_variance_values, markers=True, line_shape='linear')
        fig.update_layout(
            title='Explained Variance vs. Number of Components',
            xaxis_title='Number of Components',
            yaxis_title='Cumulative Explained Variance'
        )
        fig.show()

        # Determine the number of components based on the target variance
        num_components = np.where(cum_variance >= target_variance)[0][0] + 1
        print(f'Number of components to achieve the {target_variance} target variance = {num_components}')


class Models:
    
    @classmethod
    def perform_model(cls, df: pd.DataFrame, target: str, type_model='linear'):
        """
        Perform a regression model using Statsmodels.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        target (str): The name of the dependent variable column.
        type_model (str): The type of regression model to perform, 'linear' (default), 'logit', 'probit', 'robust'.

        Returns:
        results (statsmodels.regression.linear_model.RegressionResultsWrapper): The results of the regression.

        This class method fits a specified type of regression model to the provided data using Statsmodels.
        It supports linear, logistic, Poisson, and robust linear regression models. The results of the regression
        are printed, and the results object is returned.

        Example:
        results = RegressionModel.perform_model(df, 'DependentVariable', 'linear')
        """
        # Prepare the data
        X = df.drop(target, axis=1)
        y = df[target]

        # Add a constant to the independent variable (intercept)
        X = sm.add_constant(X)

        if type_model.lower() == 'linear':
            # Create a linear regression model
            model = sm.OLS(y, X)
        elif type_model.lower() == 'logit':
            # Create a logistic regression model
            model = sm.Logit(y, X)
        elif type_model.lower() == 'probit':
            # Create a probit regression model
            model = sm.Probit(y, X)
        elif type_model.lower() == 'robust':
            # Create a robust linear regression model
            model = sm.RLM(y, sm.add_constant(X), M=sm.robust.norms.HuberT())
        else:
            raise ValueError("Valid type_model values: 'linear' (default), 'logit', 'probit', 'robust'")

        # Fit the model to the data
        results = model.fit()

        # Print the summary of the regression results
        print(results.summary())

        return results

