"""
Script principal para análise avançada de séries temporais
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
from advanced_statistical_tests import AdvancedStatisticalTests
from advanced_models import AdvancedTimeSeriesModels
from config import TEST_SIZE, FORECAST_STEPS


def run_advanced_analysis():
    """Executa análise avançada completa"""
    print("=" * 80)
    print("ANÁLISE AVANÇADA DE SÉRIES TEMPORAIS - PRODUÇÃO ELÉTRICA")
    print("=" * 80)
    print()
    
    # 1. CARREGAMENTO DOS DADOS
    print("1. CARREGANDO DADOS...")
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    if df is None:
        print("Erro ao carregar dados. Encerrando...")
        return
    
    print(f"Dados carregados: {df.shape}")
    print()
    
    # 2. ANÁLISE EXPLORATÓRIA BÁSICA
    print("2. ANÁLISE EXPLORATÓRIA BÁSICA...")
    analyzer = DataAnalyzer(df)
    basic_stats = analyzer.calculate_basic_statistics()
    
    print("Estatísticas básicas:")
    for key, value in basic_stats.items():
        print(f"  {key}: {value}")
    print()
    
    # 3. TESTES ESTATÍSTICOS AVANÇADOS
    print("3. TESTES ESTATÍSTICOS AVANÇADOS...")
    advanced_tester = AdvancedStatisticalTests(df)
    
    # 3.1 Testes de estacionariedade avançados
    print("3.1 Testes de estacionariedade avançados...")
    stationarity_tests = advanced_tester.comprehensive_stationarity_tests()
    
    # 3.2 Testes de normalidade
    print("3.2 Testes de normalidade...")
    normality_tests = advanced_tester.normality_tests()
    
    # 3.3 Testes de heterocedasticidade
    print("3.3 Testes de heterocedasticidade...")
    heteroscedasticity_tests = advanced_tester.heteroscedasticity_tests()
    
    # 3.4 Testes de autocorrelação
    print("3.4 Testes de autocorrelação...")
    autocorrelation_tests = advanced_tester.autocorrelation_tests()
    
    # 3.5 Testes de efeitos ARCH
    print("3.5 Testes de efeitos ARCH...")
    arch_tests = advanced_tester.arch_effects_test()
    
    # 3.6 Relatório completo
    print("3.6 Gerando relatório estatístico completo...")
    statistical_report = advanced_tester.generate_comprehensive_report()
    
    print("Resumo executivo dos testes estatísticos:")
    summary = statistical_report['executive_summary']
    for key, value in summary.items():
        if key == 'recommendations':
            print(f"  {key}:")
            for rec in value:
                print(f"    - {rec}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # 4. TRANSFORMAÇÕES AVANÇADAS
    print("4. APLICANDO TRANSFORMAÇÕES AVANÇADAS...")
    transformer = DataTransformer(df)
    
    # Pipeline completo de transformações
    transformed_data = transformer.apply_full_transformation_pipeline()
    
    if transformed_data is not None:
        print("Transformações aplicadas com sucesso!")
        
        # Testar estacionariedade após transformações
        transformed_tester = AdvancedStatisticalTests(transformed_data)
        transformed_stationarity = transformed_tester.comprehensive_stationarity_tests()
        
        print("Teste de estacionariedade após transformações:")
        if 'adf' in transformed_stationarity:
            adf_tests = [r for r in transformed_stationarity['adf'].values() 
                        if isinstance(r, dict) and 'is_stationary' in r]
            if adf_tests:
                stationary_count = sum(1 for test in adf_tests if test['is_stationary'])
                total_tests = len(adf_tests)
                print(f"  Estacionários: {stationary_count}/{total_tests} ({stationary_count/total_tests*100:.1f}%)")
        print()
    else:
        print("Erro ao aplicar transformações!")
        return
    
    # 5. MODELOS BÁSICOS
    print("5. TREINANDO MODELOS BÁSICOS...")
    basic_models = TimeSeriesModels(transformed_data)
    
    # Treinar modelos básicos
    persistence_results = basic_models.persistence_model(test_size=TEST_SIZE)
    arima_results = basic_models.arima_model(test_size=TEST_SIZE)
    sarimax_results = basic_models.sarimax_model(test_size=TEST_SIZE)
    
    # 6. MODELOS AVANÇADOS
    print("6. TREINANDO MODELOS AVANÇADOS...")
    advanced_models = AdvancedTimeSeriesModels(transformed_data)
    
    # Executar todos os modelos avançados
    advanced_results = advanced_models.run_all_advanced_models()
    
    # 7. MODELO XGBOOST
    print("7. TREINANDO XGBOOST...")
    train_data, test_data = data_loader.split_by_date('2016-12-31')
    
    if train_data is not None and test_data is not None:
        xgb_results = basic_models.xgboost_model(train_data, test_data)
    else:
        xgb_results = None
    
    # 8. AVALIAÇÃO COMPREENSIVA
    print("8. AVALIAÇÃO COMPREENSIVA DOS MODELOS...")
    
    # Coletar todos os resultados
    all_results = {
        'persistence': persistence_results,
        'arima': arima_results,
        'sarimax': sarimax_results,
        'xgboost': xgb_results
    }
    
    # Adicionar resultados avançados
    for category, results in advanced_results.items():
        if isinstance(results, dict) and 'error' not in results:
            if category == 'machine_learning':
                # Para ML, usar o melhor modelo de cada categoria
                for model_name, model_data in results.items():
                    if model_name != 'best_model' and isinstance(model_data, dict):
                        all_results[f'ml_{model_name}'] = model_data
            
            elif category == 'ensemble':
                # Para ensemble, usar o melhor ensemble
                best_ensemble = results.get('best_ensemble')
                if best_ensemble:
                    all_results[f'ensemble_{best_ensemble}'] = results[best_ensemble]
            
            elif category == 'garch':
                # Para GARCH, usar o melhor modelo
                best_garch = results.get('best_model')
                if best_garch:
                    all_results[f'garch_{best_garch}'] = results[best_garch]
    
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
    
    # 9. VISUALIZAÇÕES AVANÇADAS
    print("9. GERANDO VISUALIZAÇÕES AVANÇADAS...")
    visualizer = DataVisualizer()
    
    # Gráficos básicos
    visualizer.plot_time_series(df, "Série Temporal Original")
    visualizer.plot_histogram(df, "Distribuição Original")
    
    # Estatísticas móveis
    rolling_mean, rolling_std = analyzer.calculate_rolling_statistics()
    visualizer.plot_rolling_statistics(df, rolling_mean, rolling_std, "Estatísticas Móveis")
    
    # Comparação de modelos
    visualizer.plot_model_comparison(all_results, metric='MSE')
    
    # Previsões dos modelos
    for model_name, results in all_results.items():
        if 'predictions' in results and 'actual' in results:
            visualizer.plot_model_predictions(
                results['actual'], 
                results['predictions'], 
                f"Previsões - {model_name}", 
                model_name.title()
            )
    
    # 10. RELATÓRIOS FINAIS
    print("10. GERANDO RELATÓRIOS FINAIS...")
    
    # Relatório de avaliação
    evaluation_report = evaluator.generate_evaluation_report()
    print(evaluation_report)
    
    # Relatório estatístico
    print("\n" + "=" * 80)
    print("RELATÓRIO ESTATÍSTICO COMPLETO")
    print("=" * 80)
    print(statistical_report['executive_summary'])
    
    # Relatório dos modelos avançados
    print("\n" + "=" * 80)
    print("RELATÓRIO DOS MODELOS AVANÇADOS")
    print("=" * 80)
    advanced_report = advanced_models.generate_advanced_report()
    print(advanced_report)
    
    # DataFrame de comparação
    comparison_df = evaluator.create_comparison_dataframe()
    if comparison_df is not None:
        print("\nComparação final dos modelos:")
        print(comparison_df)
    
    print("\n" + "=" * 80)
    print("ANÁLISE AVANÇADA CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
    
    return {
        'data': df,
        'transformed_data': transformed_data,
        'statistical_report': statistical_report,
        'basic_models': all_results,
        'advanced_models': advanced_results,
        'evaluation': evaluation_summary,
        'best_model': best_model
    }


def run_specific_tests():
    """Executa testes específicos"""
    print("Executando testes específicos...")
    
    # Carregar dados
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    if df is None:
        return
    
    # Testes estatísticos específicos
    advanced_tester = AdvancedStatisticalTests(df)
    
    print("\n1. Testes de estacionariedade:")
    stationarity = advanced_tester.comprehensive_stationarity_tests()
    print(f"   ADF: {len(stationarity.get('adf', {}))} especificações testadas")
    
    print("\n2. Testes de normalidade:")
    normality = advanced_tester.normality_tests()
    for test_name, test_result in normality.items():
        if isinstance(test_result, dict) and 'is_normal' in test_result:
            print(f"   {test_name}: {'Normal' if test_result['is_normal'] else 'Não Normal'}")
    
    print("\n3. Testes de heterocedasticidade:")
    hetero = advanced_tester.heteroscedasticity_tests()
    if 'error' not in hetero:
        for test_name, test_result in hetero.items():
            if isinstance(test_result, dict) and 'is_heteroscedastic' in test_result:
                print(f"   {test_name}: {'Presente' if test_result['is_heteroscedastic'] else 'Ausente'}")
    
    print("\n4. Testes de autocorrelação:")
    autocorr = advanced_tester.autocorrelation_tests()
    if 'error' not in autocorr:
        ljung_box = autocorr.get('ljung_box', {})
        if 'significant_autocorr' in ljung_box:
            print(f"   Ljung-Box: {'Significativa' if ljung_box['significant_autocorr'] else 'Não significativa'}")
    
    print("\n5. Testes de efeitos ARCH:")
    arch = advanced_tester.arch_effects_test()
    if 'error' not in arch:
        best_garch = arch.get('best_model', 'N/A')
        print(f"   Melhor modelo GARCH: {best_garch}")


if __name__ == "__main__":
    print("Escolha uma opção:")
    print("1. Análise completa avançada")
    print("2. Testes estatísticos específicos")
    
    choice = input("Digite sua escolha (1 ou 2): ").strip()
    
    if choice == "1":
        results = run_advanced_analysis()
        print(f"\nAnálise concluída! {len(results['basic_models'])} modelos treinados.")
    elif choice == "2":
        run_specific_tests()
    else:
        print("Opção inválida. Executando análise completa...")
        results = run_advanced_analysis()
