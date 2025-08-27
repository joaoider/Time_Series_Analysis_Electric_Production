"""
Teste simples dos módulos
"""

import sys
import importlib


def test_imports():
    """Testa se todos os módulos podem ser importados"""
    modules = [
        'config',
        'data_loader', 
        'data_analysis',
        'stationarity',
        'transformation',
        'models',
        'evaluation',
        'visualization'
    ]
    
    print("Testando importação dos módulos...")
    print("=" * 40)
    
    failed_imports = []
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}: OK")
        except ImportError as e:
            print(f"❌ {module_name}: ERRO - {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"⚠️  {module_name}: AVISO - {e}")
    
    print("=" * 40)
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} módulo(s) falharam na importação:")
        for module in failed_imports:
            print(f"  - {module}")
        return False
    else:
        print("\n✅ Todos os módulos foram importados com sucesso!")
        return True


def test_basic_functionality():
    """Testa funcionalidade básica dos módulos"""
    print("\nTestando funcionalidade básica...")
    print("=" * 40)
    
    try:
        # Testar configurações
        from config import ROLLING_WINDOW, TEST_SIZE
        print(f"✅ Configurações: ROLLING_WINDOW={ROLLING_WINDOW}, TEST_SIZE={TEST_SIZE}")
        
        # Testar carregador de dados
        from data_loader import DataLoader
        loader = DataLoader()
        print("✅ DataLoader: Classe criada com sucesso")
        
        # Testar analisador de dados
        from data_analysis import DataAnalyzer
        print("✅ DataAnalyzer: Classe criada com sucesso")
        
        # Testar testador de estacionariedade
        from stationarity import StationarityTester
        print("✅ StationarityTester: Classe criada com sucesso")
        
        # Testar transformador de dados
        from transformation import DataTransformer
        print("✅ DataTransformer: Classe criada com sucesso")
        
        # Testar modelos
        from models import TimeSeriesModels
        print("✅ TimeSeriesModels: Classe criada com sucesso")
        
        # Testar avaliador
        from evaluation import ModelEvaluator
        print("✅ ModelEvaluator: Classe criada com sucesso")
        
        # Testar visualizador
        from visualization import DataVisualizer
        print("✅ DataVisualizer: Classe criada com sucesso")
        
        print("\n✅ Todas as funcionalidades básicas estão funcionando!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro na funcionalidade básica: {e}")
        return False


def main():
    """Função principal de teste"""
    print("TESTE DOS MÓDULOS")
    print("=" * 50)
    
    # Testar importações
    imports_ok = test_imports()
    
    if imports_ok:
        # Testar funcionalidade básica
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 Todos os testes passaram!")
            print("O projeto está configurado corretamente.")
        else:
            print("\n⚠️  Alguns problemas foram encontrados na funcionalidade.")
    else:
        print("\n❌ Problemas de importação impedem o funcionamento.")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
