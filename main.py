"""
Script principal para análise de séries temporais de produção elétrica
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
from config import TEST_SIZE, FORECAST_STEPS


def main():
    """Função principal"""
    print("=" * 60)
    print("ANÁLISE DE SÉRIES TEMPORAIS - PRODUÇÃO ELÉTRICA")
    print("=" * 60)
    print()
    
    # 1. CARREGAMENTO DOS DADOS
    print("1. CARREGANDO DADOS...")
    data_loader = DataLoader()
    
    # Download do dataset (descomente se necessário)
    # data_loader.download_kaggle_data()
    
    # Carregar dados
    df = data_loader.load_data()
    if df is None:
        print("Erro ao carregar dados. Encerrando...")
        return
    
    print(f"Dados carregados: {df.shape}")
    print()
    
    # 2. ANÁLISE EXPLORATÓRIA
    print("2. ANÁLISE EXPLORATÓRIA...")
    analyzer = DataAnalyzer(df)
    
    # Estatísticas básicas
    basic_stats = analyzer.calculate_basic_statistics()
    print("Estatísticas básicas:")
    for key, value in basic_stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Verificação de estacionariedade por divisão
    stationarity_check = analyzer.check_stationarity_by_splitting()
    print("Verificação de estacionariedade por divisão:")
    print(f"  Diferença de média: {stationarity_check['stationarity_check']['mean_difference']:.3f}")
    print(f"  Diferença de variância: {stationarity_check['stationarity_check']['variance_difference']:.3f}")
    print(f"  Provavelmente estacionário: {stationarity_check['stationarity_check']['likely_stationary']}")
    print()
    
    # 3. VISUALIZAÇÕES INICIAIS
    print("3. GERANDO VISUALIZAÇÕES INICIAIS...")
    visualizer = DataVisualizer()
    
    # Série temporal
    visualizer.plot_time_series(df, "Electric Production Time Series")
    
    # Histograma
    visualizer.plot_histogram(df, "Electric Production Distribution")
    
    # Estatísticas móveis
    rolling_mean, rolling_std = analyzer.calculate_rolling_statistics()
    visualizer.plot_rolling_statistics(df, rolling_mean, rolling_std, "Rolling Statistics")
    print()
    
    # 4. TESTE DE ESTACIONARIEDADE
    print("4. TESTE DE ESTACIONARIEDADE...")
    stationarity_tester = StationarityTester(df)
    
    # Teste ADF
    adf_results = stationarity_tester.adfuller_test()
    print("Teste Augmented Dickey-Fuller:")
    print(f"  Estatística ADF: {adf_results['adf_statistic']}")
    print(f"  p-value: {adf_results['p_value']}")
    print(f"  É estacionário: {adf_results['is_stationary']}")
    print()
    
    # Interpretação
    print("Interpretação:")
    for interpretation in adf_results['interpretation']:
        print(f"  {interpretation}")
    print()
    
    # 5. TRANSFORMAÇÕES DOS DADOS
    print("5. APLICANDO TRANSFORMAÇÕES...")
    transformer = DataTransformer(df)
    
    # Pipeline completo de transformações
    transformed_data = transformer.apply_full_transformation_pipeline()
    
    if transformed_data is not None:
        print("Transformações aplicadas com sucesso!")
        
        # Testar estacionariedade novamente
        transformed_stationarity = StationarityTester(transformed_data)
        transformed_adf = transformed_stationarity.adfuller_test()
        
        print("Teste ADF após transformações:")
        print(f"  Estatística ADF: {transformed_adf['adf_statistic']}")
        print(f"  p-value: {transformed_adf['p_value']}")
        print(f"  É estacionário: {transformed_adf['is_stationary']}")
        print()
        
        # Visualizar comparação
        visualizer.plot_transformation_comparison(df, transformed_data, "Data Transformation")
        
        # Decomposição sazonal
        visualizer.plot_seasonal_decomposition(transformed_data, model='additive', period=12)
        
        # ACF e PACF
        visualizer.plot_acf_pacf(transformed_data, nlags=20)
    else:
        print("Erro ao aplicar transformações!")
        return
    
    # 6. MODELOS DE SÉRIES TEMPORAIS
    print("6. TREINANDO MODELOS...")
    
    # Usar dados transformados para modelos estatísticos
    models = TimeSeriesModels(transformed_data)
    
    # Modelo de persistência
    print("Treinando modelo de persistência...")
    persistence_results = models.persistence_model(test_size=TEST_SIZE)
    
    # Modelo de autoregressão
    print("Treinando modelo de autoregressão...")
    ar_results = models.autoregression_model(order=(2, 1, 0), test_size=TEST_SIZE)
    
    # Modelo de média móvel
    print("Treinando modelo de média móvel...")
    ma_results = models.moving_average_model(order=(0, 1, 2), test_size=TEST_SIZE)
    
    # Modelo ARIMA
    print("Treinando modelo ARIMA...")
    arima_results = models.arima_model(test_size=TEST_SIZE)
    
    # Modelo SARIMAX
    print("Treinando modelo SARIMAX...")
    sarimax_results = models.sarimax_model(test_size=TEST_SIZE)
    
    # AutoARIMA
    print("Treinando AutoARIMA...")
    auto_arima_results = models.auto_arima_model()
    
    # 7. MODELO XGBOOST
    print("7. TREINANDO XGBOOST...")
    
    # Usar dados originais para XGBoost
    train_data, test_data = data_loader.split_by_date('2016-12-31')
    
    if train_data is not None and test_data is not None:
        print("Treinando modelo XGBoost...")
        xgb_results = models.xgboost_model(train_data, test_data)
    else:
        print("Erro ao dividir dados para XGBoost!")
        xgb_results = None
    
    # 8. AVALIAÇÃO DOS MODELOS
    print("8. AVALIANDO MODELOS...")
    
    # Coletar resultados de todos os modelos
    all_results = {
        'persistence': persistence_results,
        'autoregression': ar_results,
        'moving_average': ma_results,
        'arima': arima_results,
        'sarimax': sarimax_results,
        'xgboost': xgb_results
    }
    
    # Remover resultados None
    all_results = {k: v for k, v in all_results.items() if v is not None}
    
    # Avaliar modelos
    evaluator = ModelEvaluator(all_results)
    evaluation_summary = evaluator.calculate_metrics()
    
    print("Resumo da avaliação:")
    for model_name, metrics in evaluation_summary.items():
        print(f"  {model_name}: MSE = {metrics['MSE']}")
    print()
    
    # Melhor modelo
    best_model = evaluator.get_best_model()
    if best_model:
        print(f"Melhor modelo: {best_model['model_name']} (MSE: {best_model['metrics']['MSE']})")
    print()
    
    # 9. VISUALIZAÇÕES FINAIS
    print("9. GERANDO VISUALIZAÇÕES FINAIS...")
    
    # Comparação de modelos
    visualizer.plot_model_comparison(all_results, metric='MSE')
    
    # Previsões dos modelos
    for model_name, results in all_results.items():
        if 'predictions' in results and 'actual' in results:
            visualizer.plot_model_predictions(
                results['actual'], 
                results['predictions'], 
                f"{results['model_type']} Predictions", 
                model_name.title()
            )
    
    # 10. RELATÓRIO FINAL
    print("10. GERANDO RELATÓRIO FINAL...")
    
    # Relatório de avaliação
    evaluation_report = evaluator.generate_evaluation_report()
    print(evaluation_report)
    
    # DataFrame de comparação
    comparison_df = evaluator.create_comparison_dataframe()
    if comparison_df is not None:
        print("\nComparação dos modelos:")
        print(comparison_df)
    
    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    main()
