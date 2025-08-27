"""
Configurações e constantes para análise de séries temporais de produção elétrica
"""

# Configurações de dados
DATA_PATH = './data/Electric_Production.csv'
KAGGLE_DATASET = 'shenba/time-series-datasets'

# Configurações de análise
ROLLING_WINDOW = 12
TEST_SIZE = 10
FORECAST_STEPS = 12

# Configurações de modelos
ARIMA_ORDER = (2, 1, 2)
SARIMAX_ORDER = (2, 1, 2)
SARIMAX_SEASONAL_ORDER = (1, 1, 2, 6)

# Configurações de XGBoost
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000
}

# Configurações de visualização
PLOT_STYLE = 'white'
FIGURE_SIZE = (10, 6)
COLORS = {
    'original': 'cornflowerblue',
    'rolling_mean': 'firebrick',
    'rolling_std': 'limegreen',
    'predictions': 'darkorange',
    'forecast': 'green',
    'confidence': 'lightgreen'
}

# Configurações de teste de estacionariedade
ADF_SIGNIFICANCE_LEVEL = 0.05
