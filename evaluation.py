"""
Módulo para avaliação e comparação dos modelos
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import colors
from config import ADF_SIGNIFICANCE_LEVEL


class ModelEvaluator:
    """Classe para avaliação e comparação dos modelos"""
    
    def __init__(self, models_results):
        self.models_results = models_results
        self.evaluation_summary = None
        
    def calculate_metrics(self):
        """Calcula métricas de avaliação para todos os modelos"""
        if not self.models_results:
            return None
            
        metrics = {}
        
        for model_name, results in self.models_results.items():
            if results and 'mse' in results:
                mse = results['mse']
                rmse = sqrt(mse)
                
                # Calcular outras métricas se disponível
                if 'predictions' in results and 'actual' in results:
                    predictions = results['predictions']
                    actual = results['actual']
                    
                    # MAE (Mean Absolute Error)
                    mae = np.mean(np.abs(np.array(predictions) - np.array(actual)))
                    
                    # MAPE (Mean Absolute Percentage Error)
                    mape = np.mean(np.abs((np.array(actual) - np.array(predictions)) / np.array(actual))) * 100
                    
                    metrics[model_name] = {
                        'MSE': round(mse, 4),
                        'RMSE': round(rmse, 4),
                        'MAE': round(mae, 4),
                        'MAPE': round(mape, 2)
                    }
                else:
                    metrics[model_name] = {
                        'MSE': round(mse, 4),
                        'RMSE': round(rmse, 4)
                    }
        
        self.evaluation_summary = metrics
        return metrics
    
    def create_comparison_dataframe(self):
        """Cria DataFrame para comparação dos modelos"""
        if not self.evaluation_summary:
            self.calculate_metrics()
            
        if not self.evaluation_summary:
            return None
            
        # Criar DataFrame
        df = pd.DataFrame(self.evaluation_summary).T
        
        # Ordenar por MSE (menor é melhor)
        df = df.sort_values('MSE', ascending=True)
        
        return df
    
    def get_model_ranking(self):
        """Retorna ranking dos modelos baseado no MSE"""
        if not self.evaluation_summary:
            self.calculate_metrics()
            
        if not self.evaluation_summary:
            return None
            
        # Criar lista de tuplas (modelo, mse) e ordenar
        ranking = [(model, metrics['MSE']) for model, metrics in self.evaluation_summary.items()]
        ranking.sort(key=lambda x: x[1])  # Ordenar por MSE
        
        return ranking
    
    def get_best_model(self):
        """Retorna o melhor modelo baseado no MSE"""
        ranking = self.get_model_ranking()
        
        if not ranking:
            return None
            
        best_model_name = ranking[0][0]
        best_model_metrics = self.evaluation_summary[best_model_name]
        
        return {
            'model_name': best_model_name,
            'metrics': best_model_metrics,
            'rank': 1
        }
    
    def get_worst_model(self):
        """Retorna o pior modelo baseado no MSE"""
        ranking = self.get_model_ranking()
        
        if not ranking:
            return None
            
        worst_model_name = ranking[-1][0]
        worst_model_metrics = self.evaluation_summary[worst_model_name]
        
        return {
            'model_name': worst_model_name,
            'metrics': worst_model_metrics,
            'rank': len(ranking)
        }
    
    def compare_models_pairwise(self):
        """Compara modelos dois a dois"""
        if not self.evaluation_summary:
            self.calculate_metrics()
            
        if not self.evaluation_summary:
            return None
            
        models = list(self.evaluation_summary.keys())
        comparisons = {}
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Evitar comparações duplicadas
                    key = f"{model1}_vs_{model2}"
                    
                    mse1 = self.evaluation_summary[model1]['MSE']
                    mse2 = self.evaluation_summary[model2]['MSE']
                    
                    improvement = ((mse1 - mse2) / mse1) * 100
                    
                    comparisons[key] = {
                        'model1': model1,
                        'model2': model2,
                        'mse1': mse1,
                        'mse2': mse2,
                        'improvement_percent': round(improvement, 2),
                        'better_model': model1 if mse1 < mse2 else model2
                    }
        
        return comparisons
    
    def generate_evaluation_report(self):
        """Gera relatório completo de avaliação"""
        if not self.evaluation_summary:
            self.calculate_metrics()
            
        if not self.evaluation_summary:
            return "Nenhum modelo disponível para avaliação"
        
        report = []
        report.append("=" * 60)
        report.append("RELATÓRIO DE AVALIAÇÃO DOS MODELOS")
        report.append("=" * 60)
        report.append("")
        
        # Resumo geral
        report.append("RESUMO GERAL:")
        report.append(f"- Total de modelos avaliados: {len(self.evaluation_summary)}")
        
        best_model = self.get_best_model()
        worst_model = self.get_worst_model()
        
        if best_model:
            report.append(f"- Melhor modelo: {best_model['model_name']} (MSE: {best_model['metrics']['MSE']})")
        if worst_model:
            report.append(f"- Pior modelo: {worst_model['model_name']} (MSE: {worst_model['metrics']['MSE']})")
        
        report.append("")
        
        # Ranking dos modelos
        report.append("RANKING DOS MODELOS (por MSE):")
        ranking = self.get_model_ranking()
        for i, (model, mse) in enumerate(ranking, 1):
            report.append(f"{i}. {model}: MSE = {mse}")
        
        report.append("")
        
        # Comparações dois a dois
        report.append("COMPARAÇÕES DOIS A DOIS:")
        comparisons = self.compare_models_pairwise()
        for comp_key, comp_data in comparisons.items():
            report.append(f"- {comp_data['model1']} vs {comp_data['model2']}:")
            report.append(f"  Melhor: {comp_data['better_model']}")
            report.append(f"  Melhoria: {comp_data['improvement_percent']}%")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def style_dataframe(self, df):
        """Aplica estilo ao DataFrame de comparação"""
        if df is None:
            return None
            
        def coloring_bg(s, min_, max_, cmap='Reds', low=0, high=0):
            color_range = max_ - min_
            norm = colors.Normalize(min_ - (color_range*low), max_ + (color_range*high))
            normed = norm(s.values)
            c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
            return ['background-color: %s' % color for color in c]
        
        # Aplicar estilo
        styled_df = df.style.apply(
            coloring_bg, 
            min_=df.min().min(), 
            max_=df.max().max(), 
            low=0.1, 
            high=0.85
        )
        
        return styled_df
