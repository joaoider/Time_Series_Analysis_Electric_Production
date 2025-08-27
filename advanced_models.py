"""
Módulo para modelos avançados de séries temporais
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.regime_switching import MarkovRegression, MarkovAutoregression
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import ADF_SIGNIFICANCE_LEVEL


class AdvancedTimeSeriesModels:
    """Classe para modelos avançados de séries temporais"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.results = {}
        
    def regime_switching_models(self, k_regimes=2):
        """Modelos de mudança de regime (Markov Switching)"""
        try:
            # Modelo de regressão com mudança de regime
            markov_reg = MarkovRegression(
                self.data['value'], 
                k_regimes=k_regimes,
                switching_variance=True
            )
            markov_result = markov_reg.fit(disp=False)
            
            # Modelo autoregressivo com mudança de regime
            markov_ar = MarkovAutoregression(
                self.data['value'], 
                k_regimes=k_regimes,
                order=1,
                switching_variance=True
            )
            markov_ar_result = markov_ar.fit(disp=False)
            
            # Comparar modelos
            best_model = 'Markov_Reg' if markov_result.aic < markov_ar_result.aic else 'Markov_AR'
            
            results = {
                'markov_regression': {
                    'model': markov_result,
                    'aic': markov_result.aic,
                    'bic': markov_result.bic,
                    'regime_probabilities': markov_result.smoothed_marginal_probabilities,
                    'regime_means': markov_result.regime_means,
                    'regime_variances': markov_result.regime_variances
                },
                'markov_autoregression': {
                    'model': markov_ar_result,
                    'aic': markov_ar_result.aic,
                    'bic': markov_ar_result.bic,
                    'regime_probabilities': markov_ar_result.smoothed_marginal_probabilities,
                    'regime_means': markov_ar_result.regime_means,
                    'regime_variances': markov_ar_result.regime_variances
                },
                'best_model': best_model
            }
            
            self.models['regime_switching'] = results
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def var_models(self, maxlags=12):
        """Modelos VAR (Vector Autoregression)"""
        try:
            # Preparar dados multivariados (se disponível)
            if len(self.data.columns) > 1:
                var_data = self.data
            else:
                # Criar variáveis defasadas para simular VAR
                var_data = pd.DataFrame()
                var_data['value'] = self.data['value']
                var_data['value_lag1'] = self.data['value'].shift(1)
                var_data['value_lag2'] = self.data['value'].shift(2)
                var_data = var_data.dropna()
            
            # Modelo VAR
            var_model = VAR(var_data)
            
            # Seleção automática de lag
            lag_order = var_model.select_order(maxlags=maxlags)
            best_lag = lag_order.aic
            
            # Ajustar modelo com melhor lag
            var_result = var_model.fit(best_lag)
            
            # Testes de diagnóstico
            diagnostics = {
                'lag_order_selection': lag_order.summary(),
                'granger_causality': var_result.test_causality('value', ['value_lag1', 'value_lag2']),
                'normality_test': var_result.test_normality(),
                'serial_correlation': var_result.test_serial_correlation()
            }
            
            results = {
                'model': var_result,
                'best_lag': best_lag,
                'aic': var_result.aic,
                'bic': var_result.bic,
                'diagnostics': diagnostics,
                'forecast': var_result.forecast(var_data.values[-best_lag:], steps=12)
            }
            
            self.models['var'] = results
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def vecm_models(self, other_series=None):
        """Modelos VECM (Vector Error Correction Model)"""
        if other_series is None:
            return {'error': 'No other series provided for VECM'}
        
        try:
            # Preparar dados para cointegração
            data_matrix = pd.concat([self.data['value'], other_series], axis=1).dropna()
            
            # Modelo VECM
            vecm_model = VECM(data_matrix, k_ar_diff=2, coint_rank=1)
            vecm_result = vecm_model.fit()
            
            results = {
                'model': vecm_result,
                'cointegration_rank': vecm_result.rank,
                'cointegration_vectors': vecm_result.beta,
                'adjustment_parameters': vecm_result.alpha,
                'aic': vecm_result.aic,
                'bic': vecm_result.bic
            }
            
            self.models['vecm'] = results
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def exponential_smoothing_models(self):
        """Modelos de suavização exponencial"""
        try:
            models = {}
            
            # Holt-Winters Aditivo
            hw_add = ExponentialSmoothing(
                self.data['value'],
                seasonal_periods=12,
                trend='add',
                seasonal='add'
            ).fit()
            
            # Holt-Winters Multiplicativo
            hw_mul = ExponentialSmoothing(
                self.data['value'],
                seasonal_periods=12,
                trend='add',
                seasonal='mul'
            ).fit()
            
            # Holt-Winters com tendência multiplicativa
            hw_mul_trend = ExponentialSmoothing(
                self.data['value'],
                seasonal_periods=12,
                trend='mul',
                seasonal='mul'
            ).fit()
            
            # Comparar modelos
            models['hw_additive'] = {
                'model': hw_add,
                'aic': hw_add.aic,
                'bic': hw_add.bic,
                'forecast': hw_add.forecast(12)
            }
            
            models['hw_multiplicative'] = {
                'model': hw_mul,
                'aic': hw_mul.aic,
                'bic': hw_mul.bic,
                'forecast': hw_mul.forecast(12)
            }
            
            models['hw_mul_trend'] = {
                'model': hw_mul_trend,
                'aic': hw_mul_trend.aic,
                'bic': hw_mul_trend.bic,
                'forecast': hw_mul_trend.forecast(12)
            }
            
            # Melhor modelo
            best_model = min(models.keys(), key=lambda x: models[x]['aic'])
            models['best_model'] = best_model
            
            self.models['exponential_smoothing'] = models
            return models
            
        except Exception as e:
            return {'error': str(e)}
    
    def garch_models(self, p=1, q=1):
        """Modelos GARCH para volatilidade"""
        try:
            # Retornos logarítmicos
            returns = np.log(self.data['value'] / self.data['value'].shift(1)).dropna()
            
            models = {}
            
            # GARCH(1,1)
            garch_11 = arch_model(returns, vol='GARCH', p=1, q=1)
            garch_11_result = garch_11.fit(disp='off')
            
            # GARCH(2,1)
            garch_21 = arch_model(returns, vol='GARCH', p=2, q=1)
            garch_21_result = garch_21.fit(disp='off')
            
            # EGARCH (Exponential GARCH)
            egarch = arch_model(returns, vol='EGARCH', p=1, q=1)
            egarch_result = egarch.fit(disp='off')
            
            # GJR-GARCH (Glosten-Jagannathan-Runkle)
            gjr_garch = arch_model(returns, vol='GARCH', p=1, q=1, o=1)
            gjr_result = gjr_garch.fit(disp='off')
            
            models['garch_11'] = {
                'model': garch_11_result,
                'aic': garch_11_result.aic,
                'bic': garch_11_result.bic,
                'parameters': garch_11_result.params.to_dict(),
                'volatility_forecast': garch_11_result.forecast(horizon=12)
            }
            
            models['garch_21'] = {
                'model': garch_21_result,
                'aic': garch_21_result.aic,
                'bic': garch_21_result.bic,
                'parameters': garch_21_result.params.to_dict(),
                'volatility_forecast': garch_21_result.forecast(horizon=12)
            }
            
            models['egarch'] = {
                'model': egarch_result,
                'aic': egarch_result.aic,
                'bic': egarch_result.bic,
                'parameters': egarch_result.params.to_dict(),
                'volatility_forecast': egarch_result.forecast(horizon=12)
            }
            
            models['gjr_garch'] = {
                'model': gjr_result,
                'aic': gjr_result.aic,
                'bic': gjr_result.bic,
                'parameters': gjr_result.params.to_dict(),
                'volatility_forecast': gjr_result.forecast(horizon=12)
            }
            
            # Melhor modelo
            best_model = min(models.keys(), key=lambda x: models[x]['aic'])
            models['best_model'] = best_model
            
            self.models['garch'] = models
            return models
            
        except Exception as e:
            return {'error': str(e)}
    
    def machine_learning_models(self, test_size=0.2):
        """Modelos de machine learning avançados"""
        try:
            # Preparar dados
            data = self.data.copy()
            
            # Criar features
            data['lag1'] = data['value'].shift(1)
            data['lag2'] = data['value'].shift(2)
            data['lag3'] = data['value'].shift(3)
            data['rolling_mean_7'] = data['value'].rolling(7).mean()
            data['rolling_std_7'] = data['value'].rolling(7).std()
            data['rolling_mean_30'] = data['value'].rolling(30).mean()
            data['rolling_std_30'] = data['value'].rolling(30).std()
            
            # Remover NaN
            data = data.dropna()
            
            # Separar features e target
            features = ['lag1', 'lag2', 'lag3', 'rolling_mean_7', 'rolling_std_7', 
                       'rolling_mean_30', 'rolling_std_30']
            X = data[features]
            y = data['value']
            
            # Divisão treino/teste
            split_idx = int(len(data) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Normalizar dados
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models = {}
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            
            models['random_forest'] = {
                'model': rf,
                'mse': mean_squared_error(y_test, rf_pred),
                'mae': mean_absolute_error(y_test, rf_pred),
                'feature_importance': dict(zip(features, rf.feature_importances_)),
                'predictions': rf_pred
            }
            
            # Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X_train_scaled, y_train)
            gb_pred = gb.predict(X_test_scaled)
            
            models['gradient_boosting'] = {
                'model': gb,
                'mse': mean_squared_error(y_test, gb_pred),
                'mae': mean_absolute_error(y_test, gb_pred),
                'feature_importance': dict(zip(features, gb.feature_importances_)),
                'predictions': gb_pred
            }
            
            # Support Vector Regression
            svr = SVR(kernel='rbf', C=100, gamma='scale')
            svr.fit(X_train_scaled, y_train)
            svr_pred = svr.predict(X_test_scaled)
            
            models['svr'] = {
                'model': svr,
                'mse': mean_squared_error(y_test, svr_pred),
                'mae': mean_absolute_error(y_test, svr_pred),
                'predictions': svr_pred
            }
            
            # Neural Network
            mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            mlp.fit(X_train_scaled, y_train)
            mlp_pred = mlp.predict(X_test_scaled)
            
            models['neural_network'] = {
                'model': mlp,
                'mse': mean_squared_error(y_test, mlp_pred),
                'mae': mean_absolute_error(y_test, mlp_pred),
                'predictions': mlp_pred
            }
            
            # Melhor modelo
            best_model = min(models.keys(), key=lambda x: models[x]['mse'])
            models['best_model'] = best_model
            
            self.models['machine_learning'] = models
            return models
            
        except Exception as e:
            return {'error': str(e)}
    
    def ensemble_models(self, base_models_results):
        """Modelos ensemble combinando múltiplos modelos"""
        try:
            # Coletar previsões de diferentes modelos
            predictions = {}
            actual_values = None
            
            for model_type, results in base_models_results.items():
                if isinstance(results, dict) and 'predictions' in results:
                    predictions[model_type] = results['predictions']
                    if actual_values is None and 'actual' in results:
                        actual_values = results['actual']
            
            if not predictions or actual_values is None:
                return {'error': 'Insufficient model results for ensemble'}
            
            # Média simples
            ensemble_mean = np.mean(list(predictions.values()), axis=0)
            
            # Média ponderada (por MSE inverso)
            weights = {}
            for model_type, preds in predictions.items():
                mse = mean_squared_error(actual_values, preds)
                weights[model_type] = 1 / mse if mse > 0 else 0
            
            # Normalizar pesos
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
                
                # Média ponderada
                ensemble_weighted = np.zeros_like(ensemble_mean)
                for model_type, preds in predictions.items():
                    ensemble_weighted += weights[model_type] * preds
            else:
                ensemble_weighted = ensemble_mean
            
            # Mediana
            ensemble_median = np.median(list(predictions.values()), axis=0)
            
            # Calcular métricas
            ensemble_results = {
                'ensemble_mean': {
                    'mse': mean_squared_error(actual_values, ensemble_mean),
                    'mae': mean_absolute_error(actual_values, ensemble_mean),
                    'predictions': ensemble_mean
                },
                'ensemble_weighted': {
                    'mse': mean_squared_error(actual_values, ensemble_weighted),
                    'mae': mean_absolute_error(actual_values, ensemble_weighted),
                    'predictions': ensemble_weighted,
                    'weights': weights
                },
                'ensemble_median': {
                    'mse': mean_squared_error(actual_values, ensemble_median),
                    'mae': mean_absolute_error(actual_values, ensemble_median),
                    'predictions': ensemble_median
                }
            }
            
            # Melhor ensemble
            best_ensemble = min(ensemble_results.keys(), 
                              key=lambda x: ensemble_results[x]['mse'])
            ensemble_results['best_ensemble'] = best_ensemble
            
            self.models['ensemble'] = ensemble_results
            return ensemble_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_all_advanced_models(self):
        """Executa todos os modelos avançados"""
        print("Executando modelos avançados...")
        
        all_results = {}
        
        # 1. Modelos de mudança de regime
        print("1. Modelos de mudança de regime...")
        all_results['regime_switching'] = self.regime_switching_models()
        
        # 2. Modelos VAR
        print("2. Modelos VAR...")
        all_results['var'] = self.var_models()
        
        # 3. Modelos de suavização exponencial
        print("3. Modelos de suavização exponencial...")
        all_results['exponential_smoothing'] = self.exponential_smoothing_models()
        
        # 4. Modelos GARCH
        print("4. Modelos GARCH...")
        all_results['garch'] = self.garch_models()
        
        # 5. Modelos de machine learning
        print("5. Modelos de machine learning...")
        all_results['machine_learning'] = self.machine_learning_models()
        
        # 6. Modelos ensemble
        print("6. Modelos ensemble...")
        all_results['ensemble'] = self.ensemble_models(all_results)
        
        return all_results
    
    def generate_advanced_report(self):
        """Gera relatório avançado de todos os modelos"""
        if not self.models:
            return "Nenhum modelo executado ainda"
        
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO AVANÇADO DE MODELOS DE SÉRIES TEMPORAIS")
        report.append("=" * 80)
        report.append("")
        
        for model_category, results in self.models.items():
            if isinstance(results, dict) and 'error' not in results:
                report.append(f"📊 {model_category.upper()}")
                report.append("-" * 40)
                
                if 'best_model' in results:
                    report.append(f"Melhor modelo: {results['best_model']}")
                
                # Métricas específicas por categoria
                if model_category == 'garch':
                    for model_name, model_data in results.items():
                        if model_name != 'best_model' and isinstance(model_data, dict):
                            report.append(f"  {model_name}: AIC={model_data['aic']:.2f}, BIC={model_data['bic']:.2f}")
                
                elif model_category == 'machine_learning':
                    for model_name, model_data in results.items():
                        if model_name != 'best_model' and isinstance(model_data, dict):
                            report.append(f"  {model_name}: MSE={model_data['mse']:.4f}, MAE={model_data['mae']:.4f}")
                
                elif model_category == 'ensemble':
                    for ensemble_type, ensemble_data in results.items():
                        if ensemble_type != 'best_ensemble' and isinstance(ensemble_data, dict):
                            report.append(f"  {ensemble_type}: MSE={ensemble_data['mse']:.4f}, MAE={ensemble_data['mae']:.4f}")
                
                report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)
