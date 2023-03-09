"""Comming soon... """
class Eda:
    """Comming soon... """
    def __init__(self, df):
        self.df = df

    def __str__(self):
        return "Coming soon..."

    def analizar_nulos(self):
        """Devuelve porcentaje de nulos en el total de datos para cada columna"""
        return self.df.isna().sum().sort_values(ascending=False)/self.df.shape[0]*100
    
    def eliminar_nulos_si(self, p=0.5):
        """Elimina las columnas con con p porciento de valores nulos"""
        nan_percentages = self.df.isna().mean()
        mask = nan_percentages >= p
        self.df = self.df.loc[:, ~mask]
    
    def graficos_categoricos(self):
        """Crea grafico de barras horizontales para cada variable categorica"""
        cat = self.df.select_dtypes('O')
        #Calculamos el número de filas que necesitamos
        from math import ceil
        filas = ceil(cat.shape[1] / 2)
        #Definimos el gráfico
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(nrows = filas, ncols = 2, figsize = (16, filas * 6))
        #Aplanamos para iterar por el gráfico como si fuera de 1 dimensión en lugar de 2
        ax = ax.flat 
        #Creamos el bucle que va añadiendo gráficos
        for cada, variable in enumerate(cat):
            cat[variable].value_counts().plot.barh(ax = ax[cada])
            ax[cada].set_title(variable, fontsize = 12, fontweight = "bold")
            ax[cada].tick_params(labelsize = 12)

    def imputar_moda_mediana(self):
        """Imputa el valor de la mediana a los NaN en variables numericas o moda en categoricas"""
        nan_percent = self.df.isna().mean()
        mask = nan_percent > 0
        cols_to_impute = list(self.df.loc[:, mask].columns)
        for i in cols_to_impute:
            try:
                me = self.df[i].median()
                self.df[i] = self.df[i].fillna(me)
            except:
                mo = str(self.df[i].mode()[0])
                self.df[i] = self.df[i].fillna(mo)

    def eliminar_unitarios(self):
        """Elimina las variables que tienen un solo valor"""
        cols = self.df.columns
        cols_to_drop = []
        for col in cols:
            if self.df[col].nunique() == 1:
                cols_to_drop.append(col)
        self.df = self.df.drop(cols_to_drop, axis=1)

    def estadisticos_numericos(self):
        """Genera daatos estadisticos de las variables numericas"""
        num = self.df.select_dtypes('number')
        #Calculamos describe
        estadisticos = num.describe().T
        #Añadimos la mediana
        estadisticos['median'] = num.median()
        #Reordenamos para que la mediana esté al lado de la media
        estadisticos = estadisticos.iloc[:,[0,1,8,2,3,4,5,6,7]]
        #Lo devolvemos
        return(estadisticos)