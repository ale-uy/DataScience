"""
Author: ale-uy
Date: 05/2023
Updated: 08/2024
Version: v2.2
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
from scipy.stats import norm, expon, lognorm, gamma, beta, chi2, uniform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from typing import Literal, Union


# Tools:
    
def _check_numeric(df: pd.DataFrame):
    """
    Check if all columns in the DataFrame are numeric or boolean.

    Parameters:
    - df (pd.DataFrame): The DataFrame to check.

    Raises:
    - ValueError: If any column in the DataFrame is not numeric or boolean.
    """
    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number) or np.issubdtype(x, np.bool_))):
        raise ValueError("All columns in the DataFrame must be numeric or boolean. Perform EDA.convert_to_numeric()")

def _check_no_nulls(df: pd.DataFrame):
    """
    Check if there are any null values in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to check.

    Raises:
    - ValueError: If any column in the DataFrame contains null values.
    """
    if df.isnull().any().any():
        raise ValueError("The DataFrame contains null values. Perform EDA.impute_missing()")
    

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
                       method: Literal['mm', 'knn', 'interpolate'] = 'mm', 
                       num_strategy: Literal['mean', 'median', 'mode'] = 'median',
                       n_neighbors: int = 5) -> pd.DataFrame:
        """
        Imputes missing values in a DataFrame using different methods.

        Parameters:
            df (pd.DataFrame): The DataFrame containing missing values.
            method (Literal['mm', 'knn', 'interpolate']): The imputation method to use. Default is 'mm'.
                'mm' (Mean/Median/Mode): Imputes median for numeric and mode for categorical variables.
                'knn' (K-Nearest Neighbors): Imputes missing values using KNNImputer.
                'interpolate': Imputes missing values using interpolation.
            num_strategy (Literal['mean', 'median', 'mode']): The strategy to use for imputing numeric columns when method='mm'.
                                                      Default is 'median'.
            n_neighbors (int): The number of nearest neighbors to use in the KNNImputer method.
                                         Only applicable if the method is 'knn'. 5 is the default.

        Returns:
            pd.DataFrame: The original DataFrame with missing values imputed.

        Raises:
            ValueError: If an invalid method is provided.

        Usage Example:
            imputed_df = EDA.impute_missing(df, method="knn", n_neighbors=5)
        """
        df_imputed = df.copy()
        numeric_cols = df_imputed.select_dtypes(include=['number'])
        non_numeric_cols = df_imputed.select_dtypes(exclude=['number'])

        if method == 'knn':
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[numeric_cols.columns] = knn_imputer.fit_transform(numeric_cols)            
            df_imputed[non_numeric_cols.columns] = non_numeric_cols.apply(lambda col: col.fillna(col.mode()[0]))

        elif method == 'mm':
            for col in numeric_cols:
                if num_strategy == 'median':
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                elif num_strategy == 'mean':
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
                elif num_strategy == 'mode':
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode().iloc[0])
            for col in non_numeric_cols:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode().iloc[0])

        elif method == 'interpolate':
            df_imputed[numeric_cols.columns] = numeric_cols.interpolate(method='linear')
            df_imputed[non_numeric_cols.columns] = non_numeric_cols.apply(lambda col: col.fillna(col.mode()[0]))

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
        _check_no_nulls(df)
        
        df_transformed = df.copy()
        numeric_columns = df.select_dtypes(include='number').columns
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
        _check_no_nulls(df[target])
        
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

        elif  method == 'undersampling':
            try:
                rus = RandomUnderSampler(random_state=random_state)
                X_resampled, y_resampled = rus.fit_resample(X, y)
            except Exception:
                rus = RandomUnderSampler(random_state=random_state, replacement=True)
                X_resampled, y_resampled = rus.fit_resample(X, y)
                print("UnderSampler with replacement used due to insufficient samples in the majority class. It is advisable to check for duplicate data.")
            finally:
                df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                                    pd.Series(y_resampled, name=target)], axis=1)

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
        print(f'Number of duplicate observations: {df.duplicated().sum()} \n')
        print(f'Percentage of duplicate observations in total observations: % \
              {100*(df.duplicated().sum())/df.shape[0]:.1f} \n')
        cleaned_df = df.drop_duplicates(keep='first')
        return cleaned_df
        
    @classmethod
    def remove_outliers(cls, df: pd.DataFrame, method: Literal["zscore", "iqr"] = "zscore", threshold=np.inf) -> pd.DataFrame:
        """
       Remove outliers from a pandas DataFrame using different methods.

       Arguments:
           df (pd.DataFrame): The data frame from which outliers will be removed.
           method (str): The method to remove outliers, can be 'zscore' (default) or 'iqr'.
           threshold (float): The threshold for considering a value as an outlier.
           TIP: For 'zscore', a common threshold is 3. For 'iqr', a common threshold is 1.5.

       Returns:
           pd.DataFrame: A new DataFrame with no outliers.
       """
        _check_no_nulls(df)
        
        numeric_df = df.select_dtypes(include='number')
        
        if method == 'zscore':
            z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
            mask = (z_scores < threshold).all(axis=1)
        elif method == 'iqr':
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = ((numeric_df >= lower_bound) & (numeric_df <= upper_bound)).all(axis=1)
        else:
            raise ValueError("Invalid method. Please use 'zscore' or 'iqr'.")

        cleaned_df = df[mask]
        num_outliers = len(df) - len(cleaned_df)
        percentage_outliers = (num_outliers / len(df)) * 100
        print(f"Outliers remove: {num_outliers} \nPercentage of outliers removed: {percentage_outliers:.2f}%")
        
        return cleaned_df

    @classmethod
    def perform_pca(cls, df, n_components='mle'):
        """
        Perform Principal Component Analysis (PCA).

        Parameters:
        - df (pd.DataFrame or array-like): The data to perform PCA on.
        - n_components (str or int): Number of components to keep. You can use 'mle' for automatic selection
          or specify an integer for a fixed number of components.
        TIP: Use Graphs_eda.pca_elbow_method_plot() to determine the number of components.

        Returns:
        - transformed_df (pd.DataFrame): The data transformed to the new feature space.
        """
        _check_numeric(df)
        _check_no_nulls(df)
        
        if df.shape[0] <= df.shape[1]:
            raise ValueError("The number of observations must be greater than the number of features.")
        
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
        if df.select_dtypes('O').empty:
            print("No categorical columns found in the DataFrame.")
            return
        
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

        Returns:
        None
        """
        if df.select_dtypes(include='number').empty:
            print("No numerical columns found in the DataFrame.")
            return
        
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
    def scatter_plot_matrix(cls, df: pd.DataFrame) -> None:
        """
        Generates a scatter plot matrix (pair plot) for numeric columns in the DataFrame using seaborn.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        numeric_columns = df.select_dtypes(include=['number']).columns
        if numeric_columns.empty:
            print("No numeric columns found in the DataFrame.")
        else:
            sns.pairplot(df, vars=numeric_columns)
            plt.show()

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
    def hierarchical_clusters_plot(cls, df: pd.DataFrame, 
                                   method: Literal["single", "complete", "average", "median", "weighted", "ward"] = "single", 
                                   metric='euclidean', save_clusters=False):
        """
        Generates a hierarchical dendrogram from a DataFrame and saves the clusters in a series.

        Parameters:
        - df (DataFrame): The pandas DataFrame containing the data for analysis.
        - method (str): The linkage algorithm to use. Options are:
            - 'single': Single linkage
            - 'complete': Complete linkage
            - 'average': Average linkage
            - 'weighted': Weighted linkage
            - 'ward': Ward's method
            - 'centroid': Centroid linkage
            - 'median': Median linkage
        - metric (str): The distance metric to use for the linkage calculation. Common metrics include:
            - 'euclidean': Euclidean distance
            - 'cityblock': Manhattan distance
            - 'cosine': Cosine distance
            - 'hamming': Hamming distance
            - 'jaccard': Jaccard distance
            - 'chebyshev': Chebyshev distance
            - 'minkowski': Minkowski distance
        - save_clusters (bool): If True, generates a series with the cluster of each observation. 
            Will prompt for the number of clusters to use.

        Returns:
        - Series: A pandas Series with the generated clusters.
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

    @classmethod
    def _plot_distribution_comparison(cls, data, distribution, params, column_name):
        """
        Plot the real distribution of the data and the fitted distribution for comparison.

        Parameters:
            data (pd.Series): The real data.
            distribution (scipy.stats distribution): The fitted distribution.
            params (tuple): The parameters of the fitted distribution.
            column_name (str): The name of the column being plotted.
        """

        # Generate the fitted distribution
        x = np.linspace(min(data), max(data), 100)
        y = distribution.pdf(x, *params)

        # Plot the real data distribution
        sns.histplot(data, kde=False, stat='density', bins=30, label='Real Data', color='blue')

        # Plot the fitted distribution
        plt.plot(x, y, label=f'Fitted {distribution.name}', color='red')

        # Add title and labels
        plt.title(f'Distribution Comparison for {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Density')
        plt.legend()

        # Show the plot
        plt.show()


class Models:
    
    @classmethod
    def perform_model(cls, df: pd.DataFrame, target: str, type_model='linear'):
        """
        Perform a regression model using Statsmodels.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data.
            target (str): The name of the dependent variable column.
            type_model (str): The type of regression model to perform, 'linear' (default), 'robust' for regression, and 'logit', 'probit' for category.

        Returns:
            results (statsmodels.regression.linear_model.RegressionResultsWrapper): The results of the regression.

        This class method fits a specified type of regression model to the provided data using Statsmodels.
        It supports linear, logistic, Poisson, and robust linear regression models. The results of the regression
        are printed, and the results object is returned.

        Example:
        results = Models.perform_model(df, 'DependentVariable', 'linear')
        """
        _check_no_nulls(df)
        _check_numeric(df)

        # Convert boolean columns to integers (0, 1)
        for column in df.select_dtypes(include=['bool']).columns:
            df[column] = df[column].astype(int)

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
            model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        else:
            raise ValueError("Valid type_model values: 'linear' (default), 'logit', 'probit', 'robust'")

        # Fit the model to the data
        results = model.fit()

        # Print the summary of the regression results
        print(results.summary())

        return results
    
    @classmethod
    def _fit_distribution(cls, data, distributions:list):
        """
        Fit a list of distributions to the data and return the best fit.

        Parameters:
        data (pd.Series or np.ndarray): The data to fit the distributions to.
        distributions (list): A list of scipy.stats distributions to fit to the data. 
                              Defaults to [norm, expon, lognorm, gamma, beta, chi2].

        Returns:
        best_distribution (scipy.stats distribution): The distribution that best fits the data.
        best_params (tuple): The parameters of the best fitting distribution.
        best_aic (float): The AIC value of the best fitting distribution.
        """
        _check_no_nulls(data)

        best_distribution = None
        best_params = None
        best_aic = np.inf

        for distribution in distributions:
            # Fit the distribution to the data
            params = distribution.fit(data)
            
            # Calculate the AIC
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            
            # Select the best fit
            if aic < best_aic:
                best_distribution = distribution
                best_params = params
                best_aic = aic

        return best_distribution, best_params, best_aic

    @classmethod
    def model_numeric_variables(cls, df: pd.DataFrame, graph: bool = True, distributions: list = None) -> pd.DataFrame:
        """
        Model each numeric variable in the DataFrame with the best fitting distribution.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the numeric variables to model.
        graph (bool): Whether to generate and display graphs comparing the real data distribution 
                      and the fitted distribution. Defaults to True.
        distributions (list): A list of scipy.stats distributions to fit to the data. 
                              Defaults to all continuous or discrete distributions in scipy.stats.


        Returns:
        pd.DataFrame: A DataFrame containing the best fitting distribution, parameters, and AIC 
                      for each numeric variable.
        """
        _check_no_nulls(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if distributions is None:
            distributions = [norm, expon, lognorm, gamma, beta, chi2, uniform]
        else:
            distributions = distributions
        results = []

        for col in numeric_cols:
            data = df[col].dropna()
            best_distribution, best_params, best_aic = Models._fit_distribution(data, distributions)
            shape_params = best_params[:-2]
            location = best_params[-2]
            scale = best_params[-1]
            results.append({
                'Column': col,
                'Best Distribution': best_distribution.name,
                'Shape Parameters': shape_params,
                'Location': location,
                'Scale': scale,
                'AIC': best_aic
            })

            if graph:
                Graphs_eda._plot_distribution_comparison(data, best_distribution, best_params, col)

        return pd.DataFrame(results)
