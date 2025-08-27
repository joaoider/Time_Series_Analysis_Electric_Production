"""
Exemplo de uso dos módulos individualmente
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from data_analysis import DataAnalyzer
from stationarity import StationarityTester
from transformation import DataTransformer
from models import TimeSeriesModels
from evaluation import ModelEvaluator
from visualization import DataVisualizer


def example_data_loading():
    """Exemplo de carregamento de dados"""
    print("=== EXEMPLO: CARREGAMENTO DE DADOS ===")
    
    loader = DataLoader()
    
    # Carregar dados
    df = loader.load_data()
    if df is None:
        print("Erro ao carregar dados!")
        return None
    
    # Informações dos dados
    info = loader.get_data_info()
    print(f"Shape dos dados: {info['shape']}")
    print(f"Período: {info['index_min']} até {info['index_max']}")
    
    # Dividir dados
    train_data, test_data = loader.split_by_date('2016-12-31')
    print(f"Treino: {len(train_data)} observações")
    print(f"Teste: {len(test_data)} observações")
    
    return df, train_data, test_data


def example_data_analysis(df):
    """Exemplo de análise exploratória"""
    print("\n=== EXEMPLO: ANÁLISE EXPLORATÓRIA ===")
    
    analyzer = DataAnalyzer(df)
    
    # Estatísticas básicas
    stats = analyzer.calculate_basic_statistics()
    print("Estatísticas básicas:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    # Verificação de estacionariedade
    stationarity_check = analyzer.check_stationarity_by_splitting()
    print(f"\nVerificação de estacionariedade:")
    print(f"  Diferença de média: {stationarity_check['stationarity_check']['mean_difference']:.3f}")
    print(f"  Diferença de variância: {stationarity_check['stationarity_check']['variance_difference']:.3f}")
    print(f"  Provavelmente estacionário: {stationarity_check['stationarity_check']['likely_stationary']}")
    
    return analyzer


def example_stationarity_testing(df):
    """Exemplo de teste de estacionariedade"""
    print("\n=== EXEMPLO: TESTE DE ESTACIONARIEDADE ===")
    
    tester = StationarityTester(df)
    
    # Teste ADF principal
    adf_results = tester.adfuller_test()
    print("Teste Augmented Dickey-Fuller:")
    print(f"  Estatística ADF: {adf_results['adf_statistic']}")
    print(f"  p-value: {adf_results['p_value']}")
    print(f"  É estacionário: {adf_results['is_stationary']}")
    
    # Interpretação
    print("\nInterpretação:")
    for interpretation in adf_results['interpretation']:
        print(f"  {interpretation}")
    
    # Resumo completo
    summary = tester.get_stationarity_summary()
    print(f"\nConclusão geral: {summary['overall_conclusion']}")
    
    return tester


def example_data_transformation(df):
    """Exemplo de transformações dos dados"""
    print("\n=== EXEMPLO: TRANSFORMAÇÕES DOS DADOS ===")
    
    transformer = DataTransformer(df)
    
    # Pipeline completo
    transformed_data = transformer.apply_full_transformation_pipeline()
    
    if transformed_data is not None:
        print("Transformações aplicadas com sucesso!")
        
        # Testar estacionariedade após transformações
        transformed_tester = StationarityTester(transformed_data)
        transformed_adf = transformed_tester.adfuller_test()
        
        print(f"ADF após transformações: {transformed_adf['adf_statistic']}")
        print(f"É estacionário após transformações: {transformed_adf['is_stationary']}")
        
        return transformed_data
    else:
        print("Erro nas transformações!")
        return None


def example_model_training(transformed_data, train_data, test_data):
    """Exemplo de treinamento de modelos"""
    print("\n=== EXEMPLO: TREINAMENTO DE MODELOS ===")
    
    # Modelos estatísticos com dados transformados
    models = TimeSeriesModels(transformed_data)
    
    # Treinar alguns modelos
    print("Treinando modelos estatísticos...")
    persistence_results = models.persistence_model(test_size=10)
    arima_results = models.arima_model(test_size=10)
    
    # Modelo XGBoost com dados originais
    print("Treinando XGBoost...")
    xgb_results = models.xgb_model(train_data, test_data)
    
    # Resumo dos modelos
    summary = models.get_all_models_summary()
    print("\nResumo dos modelos treinados:")
    for model_name, metrics in summary.items():
        print(f"  {model_name}: MSE = {metrics['mse']}")
    
    # Melhor modelo
    best_model = models.get_best_model()
    if best_model:
        print(f"\nMelhor modelo: {best_model['model_name']} (MSE: {best_model['mse']})")
    
    return models


def example_model_evaluation(models):
    """Exemplo de avaliação dos modelos"""
    print("\n=== EXEMPLO: AVALIAÇÃO DOS MODELOS ===")
    
    # Coletar resultados
    all_results = {}
    for model_name, model_obj in models.models.items():
        if model_obj and 'mse' in model_obj:
            all_results[model_name] = model_obj
    
    # Avaliar
    evaluator = ModelEvaluator(all_results)
    evaluation_summary = evaluator.calculate_metrics()
    
    print("Métricas de avaliação:")
    for model_name, metrics in evaluation_summary.items():
        print(f"  {model_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value}")
    
    # Ranking
    ranking = evaluator.get_model_ranking()
    print(f"\nRanking dos modelos:")
    for i, (model, mse) in enumerate(ranking, 1):
        print(f"  {i}. {model}: MSE = {mse}")
    
    # Relatório completo
    report = evaluator.generate_evaluation_report()
    print(f"\n{report}")
    
    return evaluator


def example_visualization(df, transformed_data, models):
    """Exemplo de visualizações"""
    print("\n=== EXEMPLO: VISUALIZAÇÕES ===")
    
    visualizer = DataVisualizer()
    
    # Gráficos básicos
    print("Gerando gráficos básicos...")
    visualizer.plot_time_series(df, "Série Temporal Original")
    visualizer.plot_histogram(df, "Distribuição Original")
    
    if transformed_data is not None:
        visualizer.plot_transformation_comparison(df, transformed_data, "Comparação de Transformações")
        visualizer.plot_seasonal_decomposition(transformed_data, model='additive', period=12)
        visualizer.plot_acf_pacf(transformed_data, nlags=20)
    
    # Gráficos dos modelos
    if models:
        print("Gerando gráficos dos modelos...")
        visualizer.plot_model_comparison(models.models, metric='MSE')
        
        # Previsões dos modelos
        for model_name, results in models.models.items():
            if results and 'predictions' in results and 'actual' in results:
                visualizer.plot_model_predictions(
                    results['actual'],
                    results['predictions'],
                    f"Previsões - {results['model_type']}",
                    model_name.title()
                )
    
    print("Visualizações geradas com sucesso!")


def main():
    """Função principal do exemplo"""
    print("EXEMPLOS DE USO DOS MÓDULOS")
    print("=" * 50)
    
    # 1. Carregamento de dados
    result = example_data_loading()
    if result is None:
        return
    df, train_data, test_data = result
    
    # 2. Análise exploratória
    analyzer = example_data_analysis(df)
    
    # 3. Teste de estacionariedade
    tester = example_stationarity_testing(df)
    
    # 4. Transformações
    transformed_data = example_data_transformation(df)
    
    # 5. Treinamento de modelos
    models = example_model_training(transformed_data, train_data, test_data)
    
    # 6. Avaliação
    evaluator = example_model_evaluation(models)
    
    # 7. Visualizações
    example_visualization(df, transformed_data, models)
    
    print("\n" + "=" * 50)
    print("EXEMPLOS CONCLUÍDOS COM SUCESSO!")
    print("=" * 50)


if __name__ == "__main__":
    main()
