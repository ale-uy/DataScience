# -*- coding: utf-8 -*-
"""
Author: ale-uy
Date: 08/2023
Updated: 09/2023
Version: v1
File: ts.py
Description: Methods for applying time series algorithms in a straightforward manner.
License: Apache License Version 2.0
"""

import itertools
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import pacf, adfuller, kpss
from scipy.stats import boxcox, yeojohnson
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.serialize import model_to_json, model_from_json
from prophet.plot import plot, plot_components


class TS:

    @classmethod
    def load_time_series(cls, file_path: str, col_dates: str, sep=",", na_values="") -> pd.DataFrame:
        """
        Load a time series from a CSV file using pandas.

        Args:
            file_path (str): Path to the CSV file.
            col_dates (str): Name of the column containing dates to be parsed as dates.
            sep (str, optional): Delimiter used in the CSV file. Default is ','.
            na_values (str, optional): Values to be considered as missing. Default is an empty string.

        Returns:
            pd.DataFrame: DataFrame containing the time series loaded from the CSV file.
        """
        time_series = pd.read_csv(file_path, parse_dates=[col_dates], sep=sep, index_col=col_dates, na_values=na_values)
        return time_series
    
    @classmethod
    def statistical_data(cls, df: pd.DataFrame, target: str):
        """
        Identify key probabilistic statistics of the time series.

        Args:
            df (pd.DataFrame): The dataframe.
            target (str): The column containing the values of the time series to analyze.

        Returns:
            dict: A dictionary containing probabilistic statistics of the time series.
        """
        series = df[target]
        stats = {
            "Mean": series.mean(),
            "Median": series.median(),
            "Standard Deviation": series.std(),
            "Minimum": series.min(),
            "Maximum": series.max(),
            "25th Percentile": np.percentile(series, 25),
            "75th Percentile": np.percentile(series, 75),
            "Interquartile Range": np.percentile(series, 75) - np.percentile(series, 25),
            "Coefficient of Variation": (series.std() / series.mean()) * 100,
            "Skewness": series.skew(),
            "Kurtosis": series.kurtosis()
        }
        return stats
    
    @classmethod
    def unit_root_tests(cls, df: pd.DataFrame, target: str, test='adf', alpha="5%"):
        """
        Conducts a unit root test on a time series to determine its stationarity.

        Args:
            df (pd.DataFrame): The dataframe.
            target (str): The column containing the values of the time series to analyze.
            test (str, optional): The type of test to perform. It can be "adf" (Augmented Dickey-Fuller),
                "kpss" (Kwiatkowski-Phillips-Schmidt-Shin), or "pp" (Phillips Perron). Default is "adf".
            alpha (str, optional): The significance level for the test. Default is "5%".
                Options: 1%, 5% and 10%.

        Returns:
            None: If the test is "adf," it displays diagnostic information and plots if the series is non-stationary.
            dict: If the test is "kpss" or "pp," it returns a dictionary with the test results.
                The dictionary contains the KPSS Statistic, p-value, Used Lags, and Critical Values (only for "kpss").
        """
        series = df[target]
        if test == 'pp':
            test_result = adfuller(series, regression="ct")
        elif test == 'kpss':
            test_result = kpss(series)
        else:
            test_result = adfuller(series)
        print('ADF Statistic: %f' % test_result[0])
        print('p-value: %f' % test_result[1])
        d = 0
        for key, value in test_result[4].items():
            print('\t%s: %.3f' % (key, value))
        if test_result[0] > test_result[4][alpha]:
            while test_result[0] > test_result[4][alpha]:
                d += 1
                print()
                print("The series is non-stationary. Differencing is needed.")
                differenced_series = np.diff(series, n=d+1)
                test_result = adfuller(differenced_series)
                print()
                print('ADF Statistic: %f' % test_result[0])
                print('p-value: %f' % test_result[1])
                for key, value in test_result[4].items():
                    print('\t%s: %.3f' % (key, value))

                ## Plot both series
                _, ax = plt.subplots(2,1)
                ax[0].plot(series, color="red")
                ax[0].set_title("Original Series")
                ax[1].plot(differenced_series, color="blue")
                ax[1].set_title("Differenced Series")
                plt.subplots_adjust(hspace=0.5)
                plt.show()

        else:
            print("The series is stationary.")
            differenced_series = series

        print('The coefficient d: ', d)

    @classmethod
    def apply_decomposition(cls, df: pd.DataFrame, target: str, seasonal_period: int, model='additive'):
        """
        Applies seasonal decomposition to a time series.

        Args:
            df (pd.DataFrame): The dataframe.
            target (str): The name of the time series to decompose.
            seasonal_period (int): The seasonality period in the time series.
            model (str, optional): Decomposition model to use: 'additive' (default) or 'multiplicative'.

        Returns:
            tuple: A tuple containing three components: trend, seasonality, and residuals.
        """
        series = df[target]
        result = seasonal_decompose(series, model=model, period=seasonal_period)
        return result.trend, result.seasonal, result.resid

    @classmethod
    def apply_differencing(cls, df: pd.DataFrame, target: str, periods=1):
        """
        Performs differencing on a time series.

        Args:
            df (pd.DataFrame): The dataframe containing the time series data.
            target (str): The name of the time series to difference.
            periods (int, optional): The number of difference periods to apply. Default is 1.

        Returns:
            pd.Series: The differenced time series.
        """
        series = df[target]

        # Apply differencing
        differenced_series = series.diff(periods=periods).dropna()

        return differenced_series

    @classmethod
    def apply_transformation(cls, df: pd.DataFrame, target: str, method='box-cox'):
        """
        Applies transformation to a time series.

        Args:
            df (pd.DataFrame): DataFrame containing the time series.
            target (str): Name of the time series to transform (column name of the values).
            method (str, optional): Options are 'box-cox' (default), 'yj' or 'yeo-johnson', 'log' or 'logarithm'.

        Returns:
            pd.Series: Transformed time series using the selected method.
        """
        series = df[target]
        if method == 'log' or method == 'logarithm':
            transformed_data = pd.DataFrame()
            transformed_data[0] = np.log(series)
        elif method == 'yj' or method == 'yeo-johnson':
            transformed_data, lambda_val = yeojohnson(series)
            print(f"The lambda value that maximizes log-likelihood is: {lambda_val:.3f}")
        else:
            transformed_data, lambda_val = boxcox(series)
            print(f"The lambda value that maximizes log-likelihood is: {lambda_val:.3f}")

        transformed_series = pd.Series(transformed_data[0], index=df.index)
    
        return transformed_series
    
    @classmethod
    def sarima_model(cls, df: pd.DataFrame, target: str, p=0, d=0, q=0, P=0, D=0, Q=0, s=0):
        """
        Fits a SARIMA (Seasonal ARIMA) model to the time series.
    
        Args:
            df (pd.DataFrame): DataFrame containing the time series.
            target (str): Name of the time series to be modeled (column name of the values).
            p (int): Order of the autoregressive component (AR).
            d (int): Order of differencing.
            q (int): Order of the moving average component (MA).
            P (int): Order of the seasonal autoregressive component (SAR).
            D (int): Order of seasonal differencing.
            Q (int): Order of the seasonal moving average component (SMA).
            s (int): Seasonal period.
    
        Returns:
            result: Results of fitting the SARIMA model.
    
        Examples:
            # Example 1: ARIMA Model
            p, d, q, P, D, Q, s = 1, 1, 1, 0, 0, 0, 0
            result = TS.sarima_model(df, "target", p, d, q, P, D, Q, s)
    
            # Example 2: SARIMA Model with monthly seasonality on annual data
            p, d, q, P, D, Q, s = 1, 1, 1, 1, 1, 1, 12
            result = TS.sarima_model(df, "target", p, d, q, P, D, Q, s)
    
            # Example 3: ARMA Model
            p, d, q, P, D, Q, s = 2, 0, 2, 0, 0, 0, 0
            result = TS.sarima_model(df, "target", p, d, q, P, D, Q, s)
        """
        series = df[target]
        model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s))
        result = model.fit()
        return result


class Graphs_ts:

    @classmethod
    def plot_autocorrelation(cls, df, value_col: str, lags=24, alpha=0.05) -> None:
        """
        Visualizes the autocorrelation and partial autocorrelation functions graphically.
    
        Args:
            df (pandas.DataFrame): DataFrame containing time series data.
            value_col (str): Name of the column containing the values to analyze.
            lags (int, optional): Number of lags to show in the autocorrelation functions. Default is 24.
            alpha (float): Significance level or confidence level for the test.
    
        Returns:
            None
        """
        y = df[value_col]
        
        # Create the figure with subplots for ACF, PACF, and SACF
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), dpi=100)
        
        # Plot the autocorrelation function
        plot_acf(y, lags=lags, ax=ax1, alpha=alpha)
        ax1.set_title(f'Autocorrelation Function for {value_col}')
        
        # Plot the partial autocorrelation function
        plot_pacf(y, lags=lags, ax=ax2, alpha=alpha, method='ywm')
        ax2.set_title(f'Partial Autocorrelation Function for {value_col}')
        
        # Perform seasonal differencing
        seasonal_diff = y.diff(periods=12).dropna()
    
        # Plot the seasonal autocorrelation function
        plot_acf(seasonal_diff, lags=lags, ax=ax3, alpha=alpha)
        ax3.set_title(f'Seasonal Autocorrelation Function for {value_col}')
    
        # Plot the seasonal partial autocorrelation function
        seasonal_pacf = [pacf(y.diff(periods=i).dropna(), nlags=lags)[i] for i in range(1, lags + 1)]
        ax4.bar(range(1, len(seasonal_pacf) + 1), seasonal_pacf)
        ax4.set_xlabel('Lag')
        ax4.set_ylabel('SACF')
        ax4.set_title(f'Seasonal Partial Autocorrelation Function for {value_col}')
        ax4.set_xticks(range(1, len(seasonal_pacf) + 1))
    
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_seasonality_trend_residuals(cls, df, value_col: str, period=12, model='additive')->None:
        """
        Visualizes seasonality, trend, and residuals in the data using additive decomposition.

        Args:
            df (pandas.DataFrame): DataFrame containing time series data.
            value_col (str): Name of the column containing the values to analyze.
            period (int, optional): Frequency of seasonality in the data. Default is 12 for monthly data.
            model (str, optional): Model 'additive' or 'multiplicative'. Default is additive.

        Returns:
            None
        """
        y = df[value_col]

        # Perform additive decomposition
        result = seasonal_decompose(y, model=model, period=period)

        # Create a figure with subplots for the components
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), dpi=100)

        # Plot the original series
        ax1.plot(y, label='Original', color='tab:blue')
        ax1.set_ylabel('Value')
        ax1.legend()

        # Plot the trend component
        ax2.plot(result.trend, label='Trend', color='tab:orange')
        ax2.set_ylabel('Trend')
        ax2.legend()

        # Plot the seasonal component
        ax3.plot(result.seasonal, label='Seasonality', color='tab:green')
        ax3.set_ylabel('Seasonality')
        ax3.legend()

        # Plot the residual component
        ax4.plot(result.resid, label='Residual', color='tab:red')
        ax4.set_xlabel('Time Index')
        ax4.set_ylabel('Residual')
        ax4.legend()

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_box_plot(cls, df, time_col: str, value_col: str, group_by='year') -> None:
        """
        Generates and displays box plots to visualize data grouped by year, month, day, etc.

        Args:
            df (pandas.DataFrame): DataFrame containing time series data.
            time_col (str): Name of the column containing the dates to analyze.
            value_col (str): Name of the column containing the values to analyze.
            group_by (str, optional): Option for grouping the data ('year', 'month', 'day', etc.).
                                      Default is 'year' to visualize by years.

        Returns:
            None
        """
        translate = {'day': 'day', 'month': 'month', 'year': 'year'}
        if group_by == 'day':
            df['day'] = df[time_col].dt.day
            group_col = 'day'
        else:
            group_col = df[time_col].dt.__getattribute__(group_by)

        fig = px.box(df, x=group_col, y=value_col, color_discrete_sequence=px.colors.qualitative.Dark2,
                     title=f'Box Plot - By {translate[group_by].capitalize()}')
        fig.update_layout(xaxis_title=translate[group_by].capitalize(), yaxis_title='Value')
        fig.show()

    @classmethod
    def plot_correlogram(cls, df, value='value', max_lag=10, title='Correlogram Plot'):
        """
        Generates and displays a correlogram plot for a time series.

        Parameters:
            df (pd.DataFrame): DataFrame containing the time series.
            value (str): Name of the column containing the time series values.
            max_lag (int): Maximum number of lags to consider in the correlogram.
            title (str): Title of the plot.

        Explanation:
            The correlogram plot shows the cross-correlations between different
            lags in the time series. It helps identify patterns of dependence between
            different lags. Correlation values are in the range [-1, 1]. A correlation
            close to 1 indicates a strong positive correlation, while a correlation
            close to -1 indicates a strong negative correlation. A correlation close
            to 0 indicates weak or no correlation.

            If there are significantly nonzero values at certain lags, it could
            indicate temporal dependence in the data. If correlations are close to
            zero for most lags, it could indicate a stochastic process.

        Example:
            Graphs_ts.plot_correlogram(df, value='value', max_lag=20, title='Time Series Correlogram')
        """
        lags = list(range(1, max_lag + 1))
        correlations = [df[value].autocorr(lag) for lag in lags]

        fig = go.Figure(data=go.Scatter(x=lags, y=correlations, mode='markers+lines'))
        fig.update_layout(title=title, xaxis_title='Lag', yaxis_title='Correlation')
        fig.show()

    @classmethod
    def plot_prophet(cls, model, forecast, plot_components=False) -> None:
        """
        Generates and displays interactive plots related to a Prophet model and its predictions.

        Parameters:
            model: The fitted Prophet model.
            forecast: The forecasted results from the model.
            plot_components (bool): The type of plot to display. Available options:
                - If True, it plots individual components (trend, seasonality, holidays).
                - If False, it plots the main forecast.

        Example:
            Graphs.plot_prophet(prophet_model, forecast_result, plot_components=True)
        """
        if plot_components:
            plot_components(model, forecast)
        else:
            plot(model, forecast)
        

class Propheta:

    @classmethod
    def load_prophet_model(cls, model_name='prophet_model'):
        """
        Loads a previously saved Prophet model from a JSON file.

        Args:
            model_name (str): Name of the Prophet model to load.

        Returns:
            Prophet: The Prophet model loaded from the file.
        """
        filename = f'{model_name}.json'
        with open(filename, 'r') as file:
            model = model_from_json(file.read())
        return model

    @classmethod
    def train_prophet_model(cls,
            df: pd.DataFrame, 
            target: str, 
            dates: str, 
            horizon='30 days',
            grid=False, 
            parallel=None,
            rolling_window=1,
            save_model=False):
        """
        Train and fit a Prophet model for time series forecasting.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the time series data.
            target (str): The name of the column containing the target values.
            dates (str): The name of the column containing the corresponding dates.
            horizon (str): The time window for future prediction. Default is '30 days'. Options:
                "days": Days.
                "hours": Hours.
                "minutes": Minutes.
                "seconds": Seconds.
                "months": Months.
                "years": Years.
            grid (bool): Whether to perform a hyperparameter grid search.
            parallel: Parallelization options for cross_validation. Options: 'processes', 'threads'.
            rolling_window (int): Size of data window to analyze in cv. Default is 1.
            save_model (bool): Whether to save the fitted model in JSON format.

        Returns:
            Prophet: The fitted Prophet model.

        Example:
            # Create a sample DataFrame
            data = {
                'date': pd.date_range(start='2023-01-01', periods=50, freq='D'),
                'value': range(50)
            }
            df = pd.DataFrame(data)

            # Train the Prophet model
            best_model = ProphetModel.train_prophet_model(df, 'value', 'date', grid=False, save_model=False)

            # Make predictions with the model
            future = best_model.make_future_dataframe(periods=10)
            forecast = best_model.predict(future)

            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
        """
        # Prepare the data in the format required by Prophet
        df_prophet = df.rename(columns={target: 'y', dates: 'ds'})

        # Define the parameter grid for hyperparameter tuning
        if grid:
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                # Additional options that can be included:
                # 'holidays_prior_scale': [0.01, 10],
                # 'seasonality_mode': ['additive', 'multiplicative'],
                # 'changepoint_range': [0.8, 0.95]
            }
        else:
            param_grid = {
                'changepoint_prior_scale': [0.05],
                'seasonality_prior_scale': [10]
            }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each set of parameters

        # Use cross-validation to evaluate all parameter combinations
        best_model = None
        for params in all_params:
            model = Prophet(**params).fit(df_prophet)  # Fit the model with the given parameters
            df_cv = cross_validation(model, horizon=horizon, parallel=parallel)
            df_p = performance_metrics(df_cv, rolling_window=rolling_window)
            rmses.append(df_p['rmse'].values[0])  # Extract the RMSE value
            if df_p['rmse'].values[0] <= min(rmses):
                best_model = model  # Store the model with the lowest RMSE

        # Display tuning results
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        print(tuning_results.sort_values(by=['rmse']))

        # Save the best model
        if save_model:
            with open('prophet_model.json', 'w') as fout:
                fout.write(model_to_json(best_model))  # Save the model

        return best_model
