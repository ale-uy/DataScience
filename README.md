[PyPi Project](https://pypi.org/project/ale-uy/)

## Module [eda.py](): Data Manipulation

The classes eda.EDA and eda.Graphs_eda are tools for performing data manipulations and visualizations in a simple and efficient manner. These classes are designed to streamline various tasks related to data processing and cleaning.

### Available Methods

#### Data Preprocessing (EDA)

1. `EDA.remove_single_value_columns(df)`: Removes variables that have only one value in a DataFrame.

2. `EDA.remove_missing_if(df, p=0.5)`: Removes columns with a percentage of missing values greater than or equal to `p` in a DataFrame.

3. `EDA.impute_missing(df, method="median", n_neighbors=None)`: Imputes missing values in a DataFrame using the median method for numerical variables and the mode method for categorical variables. The K-Nearest Neighbors (KNN) method can also be used to impute missing values.

4. `EDA.standardize_variables(df, target, method="zscore", cols_exclude=[])`: Standardizes numerical variables in a DataFrame using the "z-score" method (mean and standard deviation-based standardization). Other standardization methods such as 'minmax' and 'robust' are also available.

5. `EDA.balance_data(df, target, oversampling=True)`: Performs random sampling of data to balance classes in a binary classification problem. This helps mitigate class imbalance issues in the dataset.

6. `EDA.shuffle_data(df)`: Shuffles the data in the DataFrame randomly, which can be useful for splitting data into training and testing sets.

7. `EDA.numeric_statistics(df)`: Generates statistical data for numerical variables in the DataFrame.

8. `EDA.convert_to_numeric(df, target, method="ohe", drop_first=True)`: Performs categorical variable encoding using different methods. In addition to "ohe" (one-hot-encode), "dummy" and "label" (label-encode) methods can be selected.

9. `EDA.analyze_nulls(df)`: Returns the percentage of null values in the entire dataset for each column.

10. `EDA.perform_full_eda(df, target, cols_exclude=[], p=0.5, impute=True, imputation_method='median', n_neighbors=None, convert=True, conversion_method="ohe", standardize=False, standardization_method="zscore", balance=False, balance_oversampling=True, shuffle=False)`: Pipeline to perform various (or all) steps of the class automatically.

#### Data Visualization (Graphs_eda)

11. `Graphs_eda.categorical_plots(df)`: Creates horizontal bar charts for each categorical variable in the DataFrame.

12. `Graphs_eda.histogram_plot(df, column)`: Generates an interactive histogram for a specific column in the DataFrame.

13. `Graphs_eda.box_plot(df, column_x, column_y)`: Generates an interactive box plot for a variable y based on another variable x.

14. `Graphs_eda.scatter_plot(df, column_x, column_y)`: Generates an interactive scatter plot for two variables, x and y.

15. `Graphs_eda.hierarchical_clusters_plot(df, method='single', metric='euclidean', save_clusters=False)`: Generates a dendrogram that is useful for determining the value of k (clusters) in hierarchical clusters.

16. `Graphs_eda.correlation_heatmap_plot(df)`: Generates a correlation heatmap for the given DataFrame.

## Module [ml.py](): Data Modeling

The classes `ml.ML`, `ml.Graphs_ml`, and `ml.Tools` are a tool for performing modeling, data manipulation, and visualization of data in a simple and efficient manner. These classes are designed to facilitate various tasks related to data processing, training, and evaluation of machine learning models.

### Data Modeling

1. `ML.lightgbm_model(...)`: Uses LightGBM to predict the target variable in a DataFrame. This method supports both classification and regression problems. You can see the customizable parameters within the docstring.

2. `ML.xgboost_model(...)`: Utilizes XGBoost to predict the target variable in a DataFrame. This method is also suitable for both classification and regression problems. You can find customizable parameters within the docstring.

3. `ML.catboost_model(...)`: Employs CatBoost to predict the target variable in a DataFrame. Similar to the previous methods, it can handle both classification and regression problems. You can explore the customizable parameters within the docstring.

> *IMPORTANT*: If you pass `grid=True` as a parameter to any of these models (e.g: `model_catboost(grid=True,...)`), a **random** hyperparameter search is now conducted to reduce training times. Additionally, you can pass `n_iter=...` with the number you desire for the model to try different combinations of parameters (10 is the default option).

#### Model Evaluation

5. **Classification Metrics**: Calculates various evaluation metrics for a classification problem, such as *precision*, *recall*, *F1-score*, and the area under the ROC curve (*AUC-ROC*).

6. **Regression Metrics**: Computes various evaluation metrics for a regression problem, including mean squared error (MSE), adjusted R-squared, among others.

#### Variable Selection and Clustering

7. `Tools.feature_importance(...)`: Calculates the importance of variables based on their contribution to prediction using Random Forest with cross-validation. It employs a threshold that determines the minimum importance required to retain or eliminate a variable. You can find customizable parameters within the docstring.

8. `Tools.generate_clusters(...)`: Applies unsupervised algorithms K-Means or DBSCAN to a DataFrame and returns a series with the cluster number to which each observation belongs. You can explore customizable parameters within the docstring.

9. `Tools.generate_soft_clusters(...)`: Applies Gaussian Mixture Models (GMM) to the DataFrame to generate a table with the probabilities of each observation belonging to a specific cluster. You can find customizable parameters within the docstring.

10. `Tools.split_and_convert_data(df, target, test_size=0.2, random_state=np.random.randint(1, 1000), encode_categorical=False)`: Divides data into training and testing sets and optionally encodes categorical variables.

11. `Graphs_ml.plot_cluster(df, random_state=np.random.randint(1, 1000))`: Elbow and silhouette plot, which is essential for determining the optimal number of clusters to use in the aforementioned clustering methods.

## Module [ts.py](): Time Series Data Manipulation

The classes `ts.Ts`, `ts.Graphs_ts`, and `ts.Propheta` are powerful tools for performing modeling, manipulation, and visualization of time series data. These classes are designed to facilitate various tasks related to statistical time series data, as well as modeling and prediction.

### Available Methods

#### TS Class
Each method has its specific functionality related to the analysis and manipulation of time series data. You can use these methods to perform various tasks on time series data, including data loading, statistical analysis, stationarity tests, decomposition, differencing, transformation, and SARIMA modeling.

1. `TS.statistical_data(df, target)`: This method calculates various statistical properties of a time series, such as mean, median, standard deviation, minimum, maximum, percentiles, coefficient of variation, skewness, and kurtosis. It returns these statistics as a dictionary.

2. `TS.unit_root_tests(df, target, test='adf', alpha="5%")`: This method performs unit root tests to determine if a time series is stationary. It supports three different tests: Augmented Dickey-Fuller (ADF), Kwiatkowski-Phillips-Schmidt-Shin (KPSS), and Phillips Perron (PP). It returns diagnostic information and, if necessary, performs differencing to make the series stationary.

3. `TS.apply_decomposition(df, target, seasonal_period, model='additive')`: This method applies seasonal decomposition to a time series, separating it into trend, seasonality, and residuals. You can specify the type of decomposition (additive or multiplicative) and the seasonal period.

4. `TS.apply_differencing(df, target, periods=1)`: This method performs differencing on a time series to make it stationary. You can specify the number of periods to difference.

5. `TS.apply_transformation(df, target, method='box-cox')`: This method applies transformations to a time series. It supports three transformation methods: Box-Cox, Yeo-Johnson, and logarithmic. It returns the transformed time series.

6. `TS.sarima_model(df, target, p=0, d=0, q=0, P=0, D=0, Q=0, s=0)`: This method fits an ARIMA model to a time series by specifying the orders of the autoregressive (AR), differencing (d), and moving average (MA) components. It can also fit a SARIMA model by modifying the other four parameters: seasonal autoregressive order (P), seasonal differencing (D), seasonal moving average (Q), and seasonal periods (s). It returns the results of fitting the ARIMA/SARIMA model.

#### Class Graphs_ts
These methods are useful for exploring and understanding time series data, identifying patterns, and evaluating model assumptions. To use these methods, you should pass a pandas DataFrame containing time series data and specify the relevant columns and parameters.

7. `Graphs_ts.plot_autocorrelation(df, value_col, lags=24, alpha=0.05)`: This method visualizes the autocorrelation function (ACF), partial autocorrelation function (PACF), and seasonal ACF of a time series (SACF and SPACF). You can specify the number of lags and the significance level of the tests.
8. `Graphs_ts.plot_seasonality_trend_residuals(df, value_col, period=12, model='additive')`: This method decomposes a time series into its trend, seasonality, and residual components using an additive or multiplicative model. It then plots these components along with the original time series.
9. `Graphs_ts.plot_box_plot(df, time_col, value_col, group_by='year')`: This method generates and displays box plots to visualize data grouped by year, month, day, etc. You can specify the time column, value column, and grouping option.
10. `Graphs_ts.plot_correlogram(df, value='value', max_lag=10, title='Correlogram Plot')`: This method creates and displays a correlogram (autocorrelation plot) for a time series. It helps identify correlations between different lags in the series.
11. `Graphs_ts.plot_prophet(model, forecast, plot_components=False)`: This method generates charts related to a Prophet model and its predictions. You can choose to visualize the components (trend, seasonality) or the entire forecast.

#### Class Propheta:
12. `Propheta.load_prophet_model(model_name='prophet_model')`: This method loads a previously saved Prophet model from a JSON file. You can specify the name of the model file to load.
13. `Propheta.train_prophet_model(...)`: This method trains and fits a Prophet model for time series forecasting. You can customize the parameters as described in the docstring.

## Install

To use the classes `ML`, `EDA`, `Graphs_ml`, `Graphs_eda`, and `Tools`, simply import the class in your code (first install with pip `pip install ale-uy`):

```python
from ale_uy.eda import EDA, Graphs_eda
from ale_uy.ml import ML, Tools, Graphs_ml
from ale_uy.ts import TS, Graphs_ts, Propheta
```

## Usage Example
Here's an example of how to use the **EDA** and **ML** classes to preprocess data and train a LightGBM model for a binary classification problem (IMPORTANT: Place the **`ale_uy/`** folder with its corresponding **[eda.py]()**, **[ts.py]()**, and **[ml.py]()** files in the working directory if you didn't install via pip (`pip install ale-uy`)):

```python
# Import the ml and eda modules with their respective classes
from ale_uy.ml import ML, Tools, Graphs_ml

from ale_uy.eda import EDA, Graphs_eda

# Load the data into a DataFrame
data = pd.read_csv(...)  # Your DataFrame with the data

# Data preprocessing with the target variable named 'target'
preprocessed_data = EDA.perform_full_eda(data, target='target')

# Train the LightGBM classification model and obtain its metrics
ML.lightgbm_model(preprocessed_data, target='target', problem_type='classification')

# If the model fits our needs, we can simply save it by adding the 'save_model=True' attribute
ML.lightgbm_model(preprocessed_data, target='target', problem_type='classification', save_model=True)
# It will be saved as "lightgbm.pkl"
```
To use the saved model with new data, we will use the following code
```python

import joblib

# File path and name where the model was saved
model_filename = "model_filename.pkl"
# Load the model
loaded_model = joblib.load(model_filename)
# Now you can use the loaded model to make predictions
# Suppose you have a dataset 'X_test' for making predictions
y_pred = loaded_model.predict(X_test)
```
## Contribution
If you encounter any issues or have ideas to improve these classes, please feel free to contribute! You can do so by submitting pull requests or opening issues on the [Project Repository](https://github.com/ale-uy/DataScience).

Thank you for your interest! I hope it proves to be a useful tool for your machine learning projects. If you have any questions or need assistance, don't hesitate to ask. Good luck with your data science and machine learning endeavors!
