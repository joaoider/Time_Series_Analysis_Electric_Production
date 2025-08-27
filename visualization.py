"""
Módulo para visualizações e gráficos
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from config import COLORS, FIGURE_SIZE, PLOT_STYLE


class DataVisualizer:
    """Classe para visualizações e gráficos"""
    
    def __init__(self):
        # Configurar estilo
        sns.set_style(PLOT_STYLE)
        rcParams['figure.figsize'] = FIGURE_SIZE
        
    def plot_rolling_statistics(self, data, rolling_mean, rolling_std, title="Rolling Statistics"):
        """Plota estatísticas móveis"""
        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(data, color=COLORS['original'], label='Original')
        plt.plot(rolling_mean, color=COLORS['rolling_mean'], label='Rolling Mean')
        plt.plot(rolling_std, color=COLORS['rolling_std'], label='Rolling Std')
        plt.xlabel('Date', size=12)
        plt.ylabel('Electric Production', size=12)
        plt.legend(loc='upper left')
        plt.title(title, size=14)
        plt.tight_layout()
        plt.show()
    
    def plot_time_series(self, data, title="Time Series Plot", ylabel="Electric Production"):
        """Plota série temporal"""
        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(data['value'], color=COLORS['original'])
        plt.xlabel('Date', size=12)
        plt.ylabel(ylabel, size=12)
        plt.title(title, size=14)
        plt.tight_layout()
        plt.show()
    
    def plot_histogram(self, data, title="Histogram", bins=30):
        """Plota histograma"""
        plt.figure(figsize=FIGURE_SIZE)
        plt.hist(data['value'], color=COLORS['original'], bins=bins, alpha=0.7)
        plt.xlabel('Electric Production', size=12)
        plt.ylabel('Frequency', size=12)
        plt.title(title, size=14)
        plt.tight_layout()
        plt.show()
    
    def plot_stationarity_test(self, data, moving_mean, moving_std, title="Stationarity Test"):
        """Plota teste de estacionariedade"""
        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(data, color=COLORS['original'], label='Original')
        plt.plot(moving_mean, color=COLORS['rolling_mean'], label='Rolling Mean')
        plt.plot(moving_std, color=COLORS['rolling_std'], label='Rolling Std')
        plt.legend(loc='upper left')
        plt.title(title, size=14)
        plt.tight_layout()
        plt.show()
    
    def plot_transformation_comparison(self, original_data, transformed_data, title="Transformation Comparison"):
        """Plota comparação entre dados originais e transformados"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]*1.5))
        
        # Dados originais
        ax1.plot(original_data, color=COLORS['original'])
        ax1.set_title('Original Data', size=12)
        ax1.set_ylabel('Electric Production')
        
        # Dados transformados
        ax2.plot(transformed_data, color=COLORS['rolling_mean'])
        ax2.set_title('Transformed Data', size=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Transformed Values')
        
        plt.suptitle(title, size=14)
        plt.tight_layout()
        plt.show()
    
    def plot_seasonal_decomposition(self, data, model='additive', period=12):
        """Plota decomposição sazonal"""
        try:
            decomposition = seasonal_decompose(data, model=model, period=period)
            
            plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]*1.5))
            decomposition.plot()
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Erro na decomposição sazonal: {e}")
    
    def plot_acf_pacf(self, data, nlags=20):
        """Plota ACF e PACF"""
        try:
            auto_corr = acf(data, nlags=nlags)
            partial_auto_corr = pacf(data, nlags=nlags, method='ols')
            
            fig, axs = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0]*1.2, FIGURE_SIZE[1]))
            
            # ACF
            axs[0].plot(auto_corr)
            axs[0].axhline(y=0, linestyle='--', color=COLORS['rolling_std'])
            axs[0].axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color=COLORS['rolling_mean'])
            axs[0].axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color=COLORS['rolling_mean'])
            axs[0].set_title('Autocorrelation Function', size=12)
            axs[0].grid(True, alpha=0.3)
            
            # PACF
            axs[1].plot(partial_auto_corr)
            axs[1].axhline(y=0, linestyle='--', color=COLORS['rolling_std'])
            axs[1].axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color=COLORS['rolling_mean'])
            axs[1].axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color=COLORS['rolling_mean'])
            axs[1].set_title('Partial Autocorrelation Function', size=12)
            axs[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Erro ao plotar ACF/PACF: {e}")
    
    def plot_model_predictions(self, actual, predictions, title="Model Predictions", model_name="Model"):
        """Plota previsões vs valores reais"""
        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(actual, label='True Values', color=COLORS['original'])
        plt.plot(predictions, label='Forecasts', color=COLORS['predictions'])
        plt.title(f'{title} - {model_name}', size=14)
        plt.legend(loc='upper left')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_forecast_with_confidence(self, data, forecast_mean, confidence_intervals, title="Forecast with Confidence Intervals"):
        """Plota previsão com intervalos de confiança"""
        plt.figure(figsize=FIGURE_SIZE)
        
        # Dados históricos
        plt.plot(data.index, data.values, label='Historical Data', color=COLORS['original'])
        
        # Previsão
        plt.plot(forecast_mean.index, forecast_mean.values, label='Forecast', color=COLORS['forecast'])
        
        # Intervalos de confiança
        if confidence_intervals is not None:
            lower_bound = confidence_intervals.iloc[:, 0]
            upper_bound = confidence_intervals.iloc[:, 1]
            plt.fill_between(forecast_mean.index, lower_bound, upper_bound, 
                           color=COLORS['confidence'], alpha=0.3, label='Confidence Interval')
        
        plt.title(title, size=14)
        plt.legend(loc='upper left')
        plt.xlabel('Date')
        plt.ylabel('Electric Production')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, models_results, metric='MSE'):
        """Plota comparação entre modelos"""
        if not models_results:
            return
            
        # Extrair métricas
        model_names = []
        metric_values = []
        
        for model_name, results in models_results.items():
            if results and 'mse' in results:
                model_names.append(model_name)
                metric_values.append(results['mse'])
        
        if not model_names:
            return
            
        # Criar gráfico de barras
        plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]*0.8))
        
        bars = plt.bar(model_names, metric_values, color=COLORS['original'], alpha=0.7)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.title(f'Model Comparison - {metric}', size=14)
        plt.ylabel(metric)
        plt.xlabel('Models')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, actual, predictions, title="Residuals Analysis"):
        """Plota análise de resíduos"""
        residuals = np.array(actual) - np.array(predictions)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(FIGURE_SIZE[0]*1.2, FIGURE_SIZE[1]*1.2))
        
        # Resíduos vs tempo
        ax1.plot(residuals, color=COLORS['original'])
        ax1.set_title('Residuals vs Time')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        # Histograma dos resíduos
        ax2.hist(residuals, bins=20, color=COLORS['original'], alpha=0.7)
        ax2.set_title('Residuals Distribution')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # Resíduos vs previsões
        ax4.scatter(predictions, residuals, alpha=0.6, color=COLORS['original'])
        ax4.axhline(y=0, color=COLORS['rolling_mean'], linestyle='--')
        ax4.set_title('Residuals vs Predictions')
        ax4.set_xlabel('Predictions')
        ax4.set_ylabel('Residuals')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, size=14)
        plt.tight_layout()
        plt.show()
    
    def save_plot(self, filename, dpi=300):
        """Salva o gráfico atual"""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Gráfico salvo como: {filename}")
