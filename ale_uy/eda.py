# -*- coding: utf-8 -*-
"""
Author: ale-uy
Date: 05/2023
Updated: 10/2023
Version: v2
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
import statsmodels.formula.api as smf
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTENC



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
    def convert_to_numeric(cls, df: pd.DataFrame, target: str, method="ohe", drop_first=True):
        """
        Performs the encoding of categorical variables using different methods.
        Parameters:
            df (pandas DataFrame): The DataFrame containing the variables to be encoded.
            target (str): The name of the target column that will not be encoded.
            method (str): The encoding method to use. Valid options:
                - "dummy": Dummy encoding.
                - "ohe": One-Hot Encoding.
                - "label": LabelEncoder.
            drop_first (bool): Drops the first dummy when using "dummy" encoding. Default is True.
        Returns:
            pandas DataFrame: The original DataFrame with the categorical columns encoded, excluding the target column.
        """

        # Separate the target column 'y' from the rest of the dataset 'X'
        y = df[target]
        if not np.issubdtype(y.dtype, np.number):
            # If 'y' is not numeric, convert it using LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(df[target])
            y = pd.Series(y_encoded, name=target)  # type: ignore
        X = df.drop(target, axis=1)

        # Use the specified encoding method
        if method == "dummy":
            """
            Performs dummy encoding on categorical variables in the DataFrame.
            Parameters:
                df (pandas DataFrame): The DataFrame containing the categorical variables.
                target (str): The name of the target column that will not be encoded.
                drop_first (bool): Drops the first dummy. Default is True.
            Returns:
                pandas DataFrame: The original DataFrame with the categorical columns encoded using
                the dummy method, excluding the target column.
            """
            # Convert to dummy
            X = pd.get_dummies(X, drop_first=drop_first)
            # Concatenate the imputed DataFrame with the encoded categorical columns to DataFrame 'Xy'
            encoded_df = pd.concat([X, y], axis=1)
        elif method == "ohe":
            """
            This method performs One-Hot Encoding for all categorical columns in the DataFrame, 
            except for the column specified as 'target', which is considered the target
            variable and will not be encoded. Categorical columns are automatically identified based on their 'object' data type.
            Returns:
                pandas DataFrame: The original DataFrame with the categorical columns encoded using
                One-Hot Encoding, excluding the target column.
            """
            # Create the OneHotEncoder with 'drop' option set to 'first' to avoid the dummy variable trap (collinearity).
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            # Get automatically identified categorical columns based on their 'object' data type
            object_columns = X.select_dtypes(include=['object']).columns
            # Apply One-Hot Encoding to the selected categorical columns and generate a new DataFrame
            encoded_df = pd.DataFrame(encoder.fit_transform(X[object_columns]))
            # Assign column names to the new DataFrame based on the original feature names
            encoded_df.columns = encoder.get_feature_names_out(object_columns)
            # Drop the original categorical columns from DataFrame 'X'
            X = X.drop(object_columns, axis=1).reset_index(drop=True)
            # Reset the index of X and encoded_df to start from 1 instead of 0
            X.index = y.index  # TEMPORARY SOLUTION #
            encoded_df.index = y.index  # TEMPORARY SOLUTION #
            # Concatenate the imputed DataFrame with the encoded categorical columns to DataFrame 'Xy'
            encoded_df = pd.concat([X, encoded_df, y], axis=1)
        elif method == "label":
            """
            Performs the encoding of categorical variables using LabelEncoder.
            Returns:
                pandas DataFrame: The original DataFrame with the categorical columns encoded
                using LabelEncoder, INCLUDING the target column.
            """
            # Create a copy of the DataFrame 'df' for encoding
            encoded_df = df.copy(deep=True)
            # Get automatically identified categorical columns based on their 'object' data type
            object_columns = encoded_df.select_dtypes(include=['object']).columns
            # Create a LabelEncoder object for each categorical column and transform the data
            label_encoders = {col: LabelEncoder() for col in object_columns}
            for col in object_columns:
                encoded_df[col] = label_encoders[col].fit_transform(encoded_df[col])
        else:
            raise ValueError("The 'method' parameter must be one of: 'dummy', 'ohe', 'label'.")
        # Return the complete 'encoded_df' DataFrame with the encoded categorical columns
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
    def impute_missing(cls, df:pd.DataFrame, method='mm', n_neighbors=None):
        """
        Imputes missing values in a DataFrame using different methods.
        Parameters:
            df (pandas DataFrame): The DataFrame containing missing values.
            method (str, optional): The imputation method to use: Default method 'mm'.
                Default method is "mm" (Median/Mode):
                    - Imputes the median value for missing values in numeric variables.
                    - Imputes the mode for categorical variables.
                Method "knn" (K-Nearest Neighbors):
                    - Imputes missing values using KNNImputer, a nearest neighbors-based method.
            n_neighbors (int, optional): The number of nearest neighbors to use in the KNNImputer method.
                                         Only applicable if the method is "knn".
        Returns:
            pandas DataFrame: The original DataFrame with missing values imputed.
            
        Usage Example:
            imputed_df = Eda.impute_missing(df, method="knn", n_neighbors=5)
        """
        if method == 'knn':
            if n_neighbors is None:
                # Perform cross-validation search to find the best value of n_neighbors
                param_grid = {'n_neighbors': [i for i in range(3, 16)]}  # Values of n_neighbors to try
                knn_imputer = KNNImputer()
                grid_search = RandomizedSearchCV(knn_imputer, param_grid, cv=3, n_iter=6)
                grid_search.fit(df)
                n_neighbors_best = grid_search.best_params_['n_neighbors']
                print(f"Best value of n_neighbors found: {n_neighbors_best}")
            else:
                n_neighbors_best = n_neighbors

            # Impute missing values using KNNImputer with the best n_neighbors value
            knn_imputer = KNNImputer(n_neighbors=n_neighbors_best)
            df_imputed = knn_imputer.fit_transform(df)
            df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
        elif method == 'mm':
            """
            Impute the median value for missing values in numeric variables
            and the mode for categorical variables.
            """
            df_imputed = df.copy(deep=True)
            for col in df_imputed.columns:
                if df_imputed[col].dtype == 'object':
                    # Impute mode for missing values in a categorical column.
                    mode = df_imputed[col].mode()[0]
                    df_imputed[col].fillna(mode, inplace=True)
                else:
                    # Impute median for missing values in a numeric column.
                    median = df_imputed[col].median()
                    df_imputed[col].fillna(median, inplace=True)
        else:
            raise ValueError('methods options: "mm" and "knn"')
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
    def standardize_variables(cls, df: pd.DataFrame, target: str, method="zscore"):
        """
        Standardizes numeric variables in a DataFrame using the specified method.
        Parameters:
            df (pandas DataFrame): The DataFrame containing the variables to standardize.
            method (str): The standardization method to use. Valid options:
                - 'zscore': Standardization using Z-Score (mean 0, standard deviation 1).
                - 'minmax': Standardization using Min-Max (range 0 to 1).
                - 'robust': Robust standardization using medians and quartiles.
                Default = zscore
            cols_exclude (list, optional): Columns that we do not want to transform.
        Returns:
            pandas DataFrame: The original DataFrame with standardized numeric variables.
        """

        y = df[target]
        aux = df.drop(columns=target, axis=1)
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("The 'method' parameter must be one of: 'zscore', 'minmax', 'robust'.")
        aux[aux.select_dtypes(include='number').columns] = scaler.fit_transform(aux.select_dtypes(include='number'))
        aux = pd.concat([aux, y], axis=1)
        return aux
    
    @classmethod
    def apply_log1p_transformation(cls, df):
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
    def balance_data(cls, df: pd.DataFrame, target: str, oversampling=True):
        """
        Balances an imbalanced dataset in a classification task using the oversampling method.
        Parameters:
            df (pandas DataFrame): The DataFrame containing input variables and the target variable.
            target (str): The name of the column containing the target variable.
            oversampling (bool, optional): Uses the "SMOTENC" algorithm for oversampling if True (default), 
                if False, uses undersampling.
        Returns:
            pandas DataFrame
        Example:
            # Suppose we have a DataFrame df with imbalanced classes
            # and we want to balance the classes using the oversampling method:
            df_balanced = Eda.balance_data(df, 'target_variable')
        """
        if oversampling:
            # Separate features and target column
            X = df.drop(columns=[target])
            y = df[target]

            # Find numerical and categorical columns
            categorical_cols = X.select_dtypes(exclude=['number']).columns

            # Create an instance of SMOTE-NC
            smote_nc = SMOTENC(categorical_features=[X.columns.get_loc(col) for col in categorical_cols])

            # Apply SMOTE-NC to generate synthetic samples
            X_resampled, y_resampled = smote_nc.fit_resample(X, y)

            # Create a new DataFrame with synthetic samples
            df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target)], axis=1)

        else:
            label_encoder = LabelEncoder()
            aux = df.copy()
            aux[target] = label_encoder.fit_transform(aux[target])
            df_no = aux[aux[target] == 0]
            df_yes = aux[aux[target] == 1]
            df_no_reduced = df_no.sample(df_yes.shape[0], random_state=1)
            df_balanced = pd.concat([df_no_reduced, df_yes], axis=0)
            df_balanced = df_balanced.sample(frac=1, random_state=1)

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
    def remove_outliers(cls, df: pd.DataFrame, method='zscore', threshold=3) -> pd.DataFrame:
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
    def perform_pca(df, n_components='mle'):
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
    def numerical_plot_density(cls, df):
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
    def box_plot(cls, df: pd.Series or pd.DataFrame) -> None:
        """
        Generates a combined box plot for numerical variables in a DataFrame or a single Series.

        Parameters:
        - data (pd.Series or pd.DataFrame): The pandas DataFrame or Series containing the data.
        This method can accept either a DataFrame with numerical variables or a single Series and generate
        a combined box plot that displays the distribution of these variables in a single chart.

        Args:
        - data (pd.Series or pd.DataFrame): The pandas DataFrame or Series containing numerical variables.

        Returns:
        None
        """
        if isinstance(df, pd.Series):
            # If input is a Series, create a DataFrame with a single column
            data = pd.DataFrame(df)
        elif isinstance(df, pd.DataFrame):
            data = df.copy()
        else:
            raise ValueError("Input must be a pandas DataFrame or Series containing numerical variables.")
        # Melt the DataFrame to have all numerical variables in a single column
        df_melted = pd.melt(data.select_dtypes(include=['float', 'int']))
        # Define a custom color palette
        custom_colors = px.colors.qualitative.Plotly  # You can change this to any other palette
        # Generate a combined box plot with the custom color palette
        fig = px.box(df_melted, x='variable', y='value', color='variable', color_discrete_sequence=custom_colors)
        fig.update_layout(title='Box Plot')
        fig.show()

    @classmethod
    def scatter_plot(cls, df: pd.DataFrame, column_x: str, column_y: str) -> None:
        """
        Generates an interactive scatter plot for two variables x and y.
        Parameters:
            column_x (str): Name of variable x in the scatter plot.
            column_y (str): Name of variable y in the scatter plot.
        """
        fig = px.scatter(df, x=column_x, y=column_y)
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
    def correlation_heatmap_plot(cls, df: pd.DataFrame) -> None:
        """
        Generates a correlation heatmap for the given DataFrame.

        Args:
            df: A DataFrame with the data.

        Returns:
            None.
        """
        corr = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, linewidth=0.5, annot=True, cmap="RdBu", vmin=-1, vmax=1)

    @classmethod
    def pca_elbow_method_plot(cls, df, target_variance=0.95) -> None:
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
    def perform_model(cls, df, target, type_model='linear'):
        """
        Perform a regression model using Statsmodels.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        target (str): The name of the dependent variable column.
        type_model (str): The type of regression model to perform ('linear' (default), 'logit', 'poisson', 'robust').

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
        elif type_model.lower() == 'poisson':
            # Create a Poisson regression model
            model = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson())
        elif type_model.lower() == 'robust':
            # Create a robust linear regression model
            model = sm.RLM(y, sm.add_constant(X), M=sm.robust.norms.HuberT())
        else:
            raise ValueError("Valid type_model values: 'linear' (default), 'logit', 'poisson', 'robust'")

        # Fit the model to the data
        results = model.fit()

        # Print the summary of the regression results
        print(results.summary())

        return results

    @classmethod
    def perform_anova(df, dependent_var, group_var, covariate_vars=None):
        """
        Perform a general ANOVA or ANCOVA using Statsmodels.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        dependent_var (str): The name of the dependent variable.
        group_var (str): The name of the categorical grouping variable.
        covariate_vars (list, optional): A list of covariate variable names. If None, performs ANOVA.

        Returns:
        results (statsmodels.regression.linear_model.RegressionResultsWrapper): The results of the ANOVA or ANCOVA.

        This function performs either a general ANOVA or ANCOVA based on the presence of covariate variables.
        If 'covariate_vars' is provided, an ANCOVA is performed. If 'covariate_vars' is None, an ANOVA is performed.

        Example:
        # Perform ANOVA
        results = perform_anova(df, 'DependentVariable', 'GroupVariable')
        
        # Perform ANCOVA with multiple covariate variables
        results = perform_anova(df, 'DependentVariable', 'GroupVariable', ['Covariate1', 'Covariate2'])
        
        # Perform ANCOVA with a single covariate variable
        results = perform_anova(df, 'DependentVariable', 'GroupVariable', ['CovariateVariable'])
        """
        # Create a formula string for the model
        formula = f'{dependent_var} ~ C({group_var})'
        if covariate_vars is not None:
            formula += ' + ' + ' + '.join(covariate_vars)

        # Fit the ANOVA or ANCOVA model
        model = smf.ols(formula, data=df)
        results = model.fit()

        # Print the summary of the ANOVA or ANCOVA results
        print(results.summary())

        return results
    
    @classmethod
    def perform_diagnostic_plots(cls, model) -> None:
        """
        Perform a diagnostic analysis of a regression model.

        Args:
            model: A fitted regression model.

        This function creates diagnostic plots and tests for a regression model, including:
        - Residuals vs. Fitted Values Plot
        - Q-Q Plot of Residuals
        - Shapiro-Wilk Normality Test of Residuals
        - Breusch-Pagan Test of Homoscedasticity
        - Displaying the regression model summary.

        The diagnostic plots are displayed in a 2-column grid.

        Returns:
            None
        """
        # Residuals
        residuals = model.resid

        # Create a subplot with two columns
        _, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Residuals vs. Fitted Values Plot
        axes[0].scatter(model.fittedvalues, residuals)
        axes[0].set_xlabel("Fitted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs. Fitted Values")

        # Q-Q Plot of Residuals
        sm.qqplot(residuals, line='s', ax=axes[1])
        axes[1].set_title("Q-Q Plot of Residuals")

        # Show the plots
        plt.tight_layout()
        plt.show()

        # Shapiro-Wilk Normality Test of Residuals
        from scipy import stats
        normality_test = stats.shapiro(residuals)
        print("Shapiro-Wilk Normality Test of Residuals:")
        print("Test Statistic =", normality_test[0])
        print("P-value =", normality_test[1])

        # Breusch-Pagan Test of Homoscedasticity
        from statsmodels.stats.diagnostic import het_breuschpagan
        het_test = het_breuschpagan(residuals, model.model.exog)
        print("\nBreusch-Pagan Test of Homoscedasticity:")
        print("LM Statistic =", het_test[0])
        print("P-value =", het_test[1])

        # Print the regression model summary
        print("\nRegression Model Summary:")
        print(model.summary())

