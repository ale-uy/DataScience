# -*- coding: utf-8 -*-
"""
Autor: ale-uy
Fecha: 08/2023
Actualizado: 09/2023
Version: v1
Archivo: ts.py
Descripción: Metodos para aplicar algoritmos de serie temporal de manera sencilla.
Licencia: Apache License Version 2.0
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
    def cargar_serie(cls, file_path:str, col_dates:str, sep=",", na_values="") -> pd.DataFrame:
        """
        Carga una serie temporal desde un archivo CSV utilizando pandas.

        Args:
            file_path (str): Ruta del archivo CSV.
            col_dates (str): Nombre de la columna que contiene las fechas a ser parseadas como fechas.
            sep (str, opcional): Delimitador utilizado en el archivo CSV. Por defecto es ','.
            na_values (str, opcional): Valores que deben considerarse como faltantes. Por defecto es una cadena vacía.

        Returns:
            pd.DataFrame: DataFrame que contiene la serie temporal cargada desde el archivo CSV.
        """
        serie = pd.read_csv(file_path, parse_dates=[col_dates], sep=sep, index_col=col_dates, na_values=na_values)
        return serie
    
    @classmethod
    def datos_estadisticos(cls, df:pd.DataFrame, target:str):
        """
        Identifica los principales datos probabilísticos de la serie temporal.

        Args:
            df (pd.DataFrame): El dataframe.
            target (str): La columna que posee los valores de la serie de tiempo a analizar.

        Returns:
            dict: Un diccionario que contiene estadísticas probabilísticas de la serie temporal.
        """
        serie = df[target]
        stats = {
            "Media": serie.mean(),
            "Mediana": serie.median(),
            "Desviación Estándar": serie.std(),
            "Mínimo": serie.min(),
            "Máximo": serie.max(),
            "Percentil 25": np.percentile(serie, 25),
            "Percentil 75": np.percentile(serie, 75),
            "Rango Interquartil": np.percentile(serie, 75) - np.percentile(serie, 25),
            "Coeficiente de Variación": (serie.std() / serie.mean()) * 100,
            "Asimetría": serie.skew(),
            "Curtosis": serie.kurtosis()
        }
        return stats
    
    @classmethod
    def pruebas_raiz_unitaria(cls, df:pd.DataFrame, target:str, prueba='adf', alpha="5%"):
        """
        Realiza una prueba de raíz unitaria en una serie de tiempo para determinar su estacionariedad.

        Args:
            df (pd.DataFrame): El dataframe.
            target (str): La columna que posee los valores de la serie de tiempo a analizar.
            prueba (str, opcional): El tipo de prueba a realizar. Puede ser "adf" (Dickey Fuller aumentada),
                "kpss" (Kwiatkowski-Phillips-Schmidt-Shin) o "pp" (Phillips Perron). Por defecto, es "adf".
            alpha (str, opcional): El nivel de significancia para la prueba. Por defecto, es "5%".

        Returns:
            None: Si la prueba es "adf" y se muestra información diagnóstica y gráficas en caso de no ser estacionaria.
            dict: Si la prueba es "kpss" o "pp", se devuelve un diccionario con los resultados de la prueba.
                El diccionario contiene el Estadístico KPSS, Valor p, Lags Usados y Valores Críticos (solo para "kpss").
        """
        serie = df[target]
        if prueba=='pp':
            test = adfuller(serie, regression="ct")
        elif prueba=='kpss':
            test = kpss(serie)
        else:
            test = adfuller(serie)
        print('ADF Statistic: %f' % test[0])
        print('p-value: %f' % test[1])
        d = 0
        for key, value in test[4].items(): # type: ignore
            print('\t%s: %.3f' % (key, value))
        if test[0] > test[4][alpha]: # type: ignore
          while test[0] > test[4][alpha]: # type: ignore
            d += 1
            print()
            print("La serie no es estacionaria. Se necesita diferenciar la serie.")
            dtrain = np.diff(serie, n=d+1)
            test = adfuller(dtrain)
            print()
            print('ADF Statistic: %f' % test[0])
            print('p-value: %f' % test[1])
            for key, value in test[4].items(): # type: ignore
                print('\t%s: %.3f' % (key, value))

            ## Graficas de ambos
            _, ax = plt.subplots(2,1)
            ax[0].plot(serie, color="red")
            ax[0].set_title("Serie original")
            ax[1].plot(dtrain, color="blue")
            ax[1].set_title("Serie diferenciada")
            plt.subplots_adjust(hspace=0.5)
            plt.show()

        else:
          print("La serie es estacionaria.")
          dtrain = serie

        print('El coeficiente d: ', d)

    @classmethod
    def aplicar_descomposicion(cls, df:pd.DataFrame, target:str, periodo_estacional:int, model='additive'):
        """
        Aplica una descomposición estacional a una serie de tiempo.

        Args:
            df (pd.DataFrame): El dataframe.
            target (str):  El nombre de la serie de tiempo a descomponer.
            periodo_estacional (int): El período de estacionalidad en la serie de tiempo.
            model (str, opcional): Modelo de descomposición a utilizar: 'additive' (defecto) o 'multiplicative'.

        Returns:
            tuple: Tupla que contiene tres componentes: tendencia, estacionalidad y residuos.
        """
        serie = df[target]
        result = seasonal_decompose(serie, model=model, period=periodo_estacional)
        return result.trend, result.seasonal, result.resid

    @classmethod
    def aplicar_diferenciacion(cls, df:pd.DataFrame, target:str, periodos=1):
        """
        Realiza la diferenciación de una serie de tiempo.

        Args:
            df (pd.DataFrame): El dataframe que contiene los datos de la serie de tiempo.
            target (str): El nombre de la serie de tiempo a diferenciar.
            periodos (int, opcional): El número de períodos de diferencia a aplicar. Por defecto, es 1.

        Returns:
            pd.Series: La serie de tiempo diferenciada.
        """
        serie = df[target]

        # Aplicar la diferenciación
        differenced_series = serie.diff(periods=periodos).dropna()

        return differenced_series

    @classmethod
    def aplicar_transformacion(cls, df:pd.DataFrame, target:str, method='box-cox'):
        """
        Aplica transformación a una serie temporal.

        Args:
            df (pd:DataFrame): dataframe que contiene la serie temporal.
            target (str): Nombre de la serie temporal a transformar (nombre de la columna valores).
            method (str, opcional): Opciones 'box-cox' (defecto), 'yj' o 'yeo-johnson', 'log' o 'logaritmo'.
                                    
        Returns:
            pd.Series: Serie temporal transformada con el método seleccionado.
        """
        serie = df[target]
        if method == 'log' or method == 'logaritmo':
            transformed_data = pd.DataFrame()
            transformed_data[0] = np.log(serie)
        elif method == 'yj' or method == 'yeo-johnson':
            transformed_data = yeojohnson(serie)
            print(f'El valor de Lambda que maximiza la log-verosimilitud es: {transformed_data[1]:.3}')
        else:
            transformed_data = boxcox(serie)
            print(f'El valor de Lambda que maximiza la log-verosimilitud es: {transformed_data[1]:.3}')
        
        transformed_serie = pd.Series(transformed_data[0], index=df.index)

        return transformed_serie

    @classmethod
    def modelo_sarima(cls, df:pd.DataFrame, target:str, p=0, d=0, q=0, P=0, D=0, Q=0, s=0):
        """
        Ajusta un modelo SARIMA (Seasonal ARIMA) a la serie temporal.

        Args:
            df (pd:DataFrame): dataframe que contiene la serie temporal.
            target (str): Nombre de la serie temporal a ser modelada (nombre de la columna valores).
            p (int): Orden del componente autoregresivo (AR).
            d (int): Orden de diferenciación.
            q (int): Orden del componente de media móvil (MA).
            P (int): Orden del componente estacional autoregresivo (SAR).
            D (int): Orden de diferenciación estacional.
            Q (int): Orden del componente estacional de media móvil (SMA).
            s (int): Período de estacionalidad.

        Returns:
            result: Resultados del ajuste del modelo SARIMA.

        Examples:
            # Ejemplo 1: Modelo ARIMA
            p, d, q, P, D, Q, s = 1, 1, 1, 0, 0, 0, 0
            result = TS.modelo_sarima(df, "target", p, d, q, P, D, Q, s)

            # Ejemplo 2: Modelo SARIMA con estacionalidad mensual en datos anuales
            p, d, q, P, D, Q, s = 1, 1, 1, 1, 1, 1, 12
            result = TS.modelo_sarima(df, "target", p, d, q, P, D, Q, s)

            # Ejemplo 3: Modelo ARMA
            p, d, q, P, D, Q, s = 2, 0, 2, 0, 0, 0, 0
            result = TS.modelo_sarima(df, "target", p, d, q, P, D, Q, s)
        """
        serie = df[target]
        model = SARIMAX(serie, order=(p, d, q), seasonal_order=(P, D, Q, s))
        result = model.fit()
        return result


class Graphs_ts:

    @classmethod
    def graficar_autocorrelacion(cls, df, value_col: str, lags=24, alpha=0.05) -> None:
        """
        Visualiza gráficamente la función de autocorrelación y autocorrelación parcial.

        Args:
            df (pandas.DataFrame): DataFrame que contiene los datos de la serie temporal.
            value_col (str): Nombre de la columna que contiene los valores a analizar.
            lags (int, opcional): Número de retrasos (lags) para mostrar en las funciones de autocorrelación.
                                  Por defecto es 24.
            alpha (float): Nivel de significancia o al nivel de confianza de la prueba.

        Returns:
            None
        """
        y = df[value_col]
        
        # Crear la figura con subplots para ACF, PACF y SACF
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), dpi=100)
        
        # Graficar la función de autocorrelación
        plot_acf(y, lags=lags, ax=ax1, alpha=alpha)
        ax1.set_title(f'Función de Autocorrelación para {value_col}')
        
        # Graficar la función de autocorrelación parcial
        plot_pacf(y, lags=lags, ax=ax2, alpha=alpha, method='ywm')
        ax2.set_title(f'Función de Autocorrelación Parcial para {value_col}')
        
        # Realizar la diferenciación estacional
        seasonal_diff = y.diff(periods=12).dropna()

        # Graficar la función de autocorrelación estacional
        plot_acf(seasonal_diff, lags=lags, ax=ax3, alpha=alpha)
        ax3.set_title(f'Función de Autocorrelación Estacional para {value_col}')


        # Graficar la función de autocorrelación parcial estacional
        seasonal_pacf = [pacf(y.diff(periods=i).dropna(), nlags=lags)[i] for i in range(1, lags + 1)]
        ax4.bar(range(1, len(seasonal_pacf) + 1), seasonal_pacf)
        ax4.set_xlabel('Lag')
        ax4.set_ylabel('SACF')
        ax4.set_title(f'Función de Autocorrelación Parcial Estacional para {value_col}')
        ax4.set_xticks(range(1, len(seasonal_pacf) + 1))

        plt.tight_layout()
        plt.show()

    @classmethod
    def graficar_estacionalidad_tendencia_ruido(cls, df, value_col: str, period=12, model='additive')->None:
        """
        Visualiza la estacionalidad, tendencia y ruido en los datos utilizando la descomposición aditiva.

        Args:
            df (pandas.DataFrame): DataFrame que contiene los datos de la serie temporal.
            value_col (str): Nombre de la columna que contiene los valores a analizar.
            freq (int, opcional): Frecuencia de la estacionalidad en los datos. Por defecto es 12 para datos mensuales.
            model (str, opcional): Modelo 'additive' o 'multiplicative'. Por defecto es additive

        Returns:
            None
        """
        y = df[value_col]
        
        # Realizar la descomposición aditiva
        result = seasonal_decompose(y, model=model, period=period)
        
        # Crear la figura con subplots para las componentes
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), dpi=100)
        
        # Graficar la serie original
        ax1.plot(y, label='Original', color='tab:blue')
        ax1.set_ylabel('Valor')
        ax1.legend()
        
        # Graficar la componente de tendencia
        ax2.plot(result.trend, label='Tendencia', color='tab:orange')
        ax2.set_ylabel('Tendencia')
        ax2.legend()
        
        # Graficar la componente de estacionalidad
        ax3.plot(result.seasonal, label='Estacionalidad', color='tab:green')
        ax3.set_ylabel('Estacionalidad')
        ax3.legend()
        
        # Graficar la componente de residuo
        ax4.plot(result.resid, label='Residuo', color='tab:red')
        ax4.set_xlabel('Índice de tiempo')
        ax4.set_ylabel('Residuo')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

    @classmethod
    def graficar_diagrama_caja(cls, df, time_col: str, value_col: str, group_by='year') -> None:
        """
        Genera y muestra diagramas de caja para visualizar los datos agrupados por año, mes o día.

        Args:
            df (pandas.DataFrame): DataFrame que contiene los datos de la serie temporal.
            time_col (str): Nombre de la columna que contiene las fechas a analizar.
            value_col (str): Nombre de la columna que contiene los valores a analizar.
            group_by (str, opcional): Opción para agrupar los datos ('year', 'month', 'day', etc.).
                                      Por defecto es 'year' para visualizar por años.

        Returns:
            None
        """
        translate = {'day':'dia','month':'mes','year':'año'}
        if group_by == 'day':
            df['day'] = df[time_col].dt.day
            group_col = 'day'
        else:
            group_col = df[time_col].dt.__getattribute__(group_by)

        fig = px.box(df, x=group_col, y=value_col, color_discrete_sequence=px.colors.qualitative.Dark2,
                    title=f'Diagrama de Caja - Por {translate[group_by].capitalize()}')
        fig.update_layout(xaxis_title=translate[group_by].capitalize(), yaxis_title='Valor')
        fig.show()

    @classmethod
    def graficar_correlograma(cls, df, value='value', max_lag=10, title='Gráfico de correlograma'):
        """
        Genera y muestra un gráfico de correlograma para una serie temporal.

        Parámetros:
            df (pd.DataFrame): DataFrame que contiene la serie temporal.
            value (str): Nombre de la columna que contiene los valores de la serie temporal.
            max_lag (int): Número máximo de retrasos a considerar en el correlograma.
            title (str): Título del gráfico.

        Explicación:
            El gráfico de correlograma muestra las correlaciones cruzadas entre diferentes
            retrasos en la serie temporal. Ayuda a identificar patrones de dependencia entre
            distintos retrasos. Los valores de correlación se encuentran en el rango [-1, 1].
            Una correlación cercana a 1 indica una fuerte correlación positiva, mientras que
            una correlación cercana a -1 indica una fuerte correlación negativa. Una correlación
            cercana a 0 indica una correlación débil o nula.

            Si hay valores significativamente diferentes de cero en los retrasos, podría
            indicar una dependencia temporal en los datos. Si las correlaciones están cerca de
            cero para la mayoría de los retrasos, podría indicar un proceso estocástico.

        Ejemplo:
            Graphs.graficar_correlograma(df, value='value', max_lag=20, title='Correlograma de la Serie Temporal')
        """
        lags = list(range(1, max_lag + 1))
        correlations = [df[value].autocorr(lag) for lag in lags]

        fig = go.Figure(data=go.Scatter(x=lags, y=correlations, mode='markers+lines'))
        fig.update_layout(title=title, xaxis_title='Lag', yaxis_title='Correlation')
        fig.show()

    @classmethod
    def graficar_profeta(cls, model, predict, graficar_componentes=False) -> None:
        """
        Genera y muestra gráficos interactivos relacionados con un modelo Prophet y sus predicciones.

        Parámetros:
            model: El modelo Prophet ajustado.
            predict: Los resultados de las predicciones del modelo.
            graficar_componentes (bool): El tipo de gráfico a mostrar. Opciones disponibles:
        """
        if graficar_componentes:
            plot_components(model, predict)
        else:
            plot(model, predict)
        

class Profeta:

    @classmethod
    def cargar_modelo_prophet(cls, name_model='prophet_model'):
        """
        Carga un modelo Prophet previamente guardado desde un archivo JSON.

        Args:
            name_model (str): Nombre del modelo prophet a cargar

        Returns:
            Prophet: El modelo Prophet cargado desde el archivo.
        """
        name = f'{name_model}.json'
        with open(name, 'r') as fin:
            model = model_from_json(fin.read())
        return model

    @classmethod
    def entrenar_modelo(cls, 
            df: pd.DataFrame, 
            target: str, 
            dates: str, 
            horizon = '30 days',
            grid = False, 
            parallel = None,
            rolling_window=1,
            save_model = False):
        """
        Entrena y ajusta un modelo Prophet para pronóstico de series temporales.

        Parámetros:
            df (pd.DataFrame): El DataFrame que contiene los datos de la serie temporal.
            target (str): El nombre de la columna que contiene los valores objetivo.
            dates (str): El nombre de la columna que contiene las fechas correspondientes.
            horizon (str): La ventana de tiempo para la predicción futura. Por defecto es '30 days'. Opciones:
                "days": Días.
                "hours": Horas.
                "minutes": Minutos.
                "seconds": Segundos.
                "months": Meses.
                "years": Años.
            grid (bool): Indica si se debe realizar una búsqueda de cuadrícula de hiperparámetros.
            parallel: Opciones de paralelización para cross_validation. Opciones: 'processes', 'threads'.
            rolling_window (int): Venta de datos que se analizan en cv. Defecto es 1.
            save_model (bool): Indica si se debe guardar el modelo ajustado en formato JSON.

        Retorna:
            Prophet: El modelo Prophet ajustado.

        Ejemplo:
            # Crear un DataFrame de ejemplo
            data = {
                'fecha': pd.date_range(start='2023-01-01', periods=50, freq='D'),
                'valor': range(50)
            }
            df = pd.DataFrame(data)

            # Entrenar el modelo Prophet
            best_model = Profeta.entrenar_modelo(df, 'target', 'dates', grid=False, save_model=False)

            # Hacer predicciones con el modelo
            future = best_model.make_future_dataframe(periods=10)
            forecast = best_model.predict(future)

            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

        """
        # Preparar los datos en el formato requerido por Prophet
        df_prophet = df.rename(columns={target: 'y', dates: 'ds'})

        # Definir la cuadrícula de parámetros para la búsqueda
        if grid:
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                # Algunas opciones que pueden incluír:
                #'holidays_prior_scale': [0.01, 10],
                #'seasonality_mode': ['additive', 'multiplicative'],
                #'changepoint_range': [0.8, 0.95]
            }
        else:
            param_grid = {
                'changepoint_prior_scale': [0.05],
                'seasonality_prior_scale': [10]
            }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        best_model = None
        for params in all_params:
            model = Prophet(**params).fit(df_prophet)  # Fit model with given params
            df_cv = cross_validation(model, horizon=horizon, parallel=parallel)
            df_p = performance_metrics(df_cv, rolling_window=rolling_window)
            rmses.append(df_p['rmse'].values[0]) # type: ignore
            if df_p['rmse'].values[0] <= min(rmses): # type: ignore
                best_model = model

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        print(tuning_results.sort_values(by=['rmse']))

        #best_params = all_params[np.argmin(rmses)]
        #print(best_params)

        if save_model:
            with open('prophet_model.json', 'w') as fout:
                fout.write(model_to_json(best_model))  # Save model

        return best_model

