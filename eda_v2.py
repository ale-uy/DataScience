import numpy as np
import matplotlib.pyplot as plt

class Eda:
    def __init__(self, df):
        """
        Inicializa una instancia de la clase Eda.

        Parameters:
            df (DataFrame): El DataFrame sobre el cual realizar el análisis exploratorio de datos.
        """
        self.df = df

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
        fig, ax = plt.subplots(nrows=rows, ncols=2, figsize=(16, rows * 6))
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
