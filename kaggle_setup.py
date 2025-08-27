"""
Script para configuração da API do Kaggle
"""

import os
import json
from pathlib import Path


def setup_kaggle():
    """Configura a API do Kaggle"""
    print("=== CONFIGURAÇÃO DA API DO KAGGLE ===")
    
    # Verificar se já existe configuração
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("Configuração do Kaggle já existe!")
        return True
    
    print("Para usar a API do Kaggle, você precisa:")
    print("1. Criar uma conta no Kaggle (https://www.kaggle.com)")
    print("2. Ir em 'Account' -> 'Create New API Token'")
    print("3. Baixar o arquivo kaggle.json")
    print("4. Colocar o arquivo na pasta .kaggle do seu usuário")
    
    # Criar diretório se não existir
    kaggle_dir.mkdir(exist_ok=True)
    
    # Solicitar credenciais
    print("\nDigite suas credenciais do Kaggle:")
    username = input("Username: ").strip()
    key = input("API Key: ").strip()
    
    if username and key:
        # Criar arquivo de configuração
        config = {
            "username": username,
            "key": key
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(config, f)
        
        # Definir permissões (apenas para Unix/Linux/Mac)
        try:
            os.chmod(kaggle_json, 0o600)
            print("Permissões definidas para o arquivo de configuração")
        except:
            print("Aviso: Não foi possível definir permissões (Windows)")
        
        print(f"Configuração salva em: {kaggle_json}")
        print("Configuração do Kaggle concluída!")
        return True
    else:
        print("Credenciais inválidas. Configuração cancelada.")
        return False


def test_kaggle_connection():
    """Testa a conexão com a API do Kaggle"""
    try:
        import kaggle
        kaggle.api.authenticate()
        print("Conexão com Kaggle estabelecida com sucesso!")
        return True
    except Exception as e:
        print(f"Erro na conexão com Kaggle: {e}")
        return False


def main():
    """Função principal"""
    print("CONFIGURAÇÃO DA API DO KAGGLE")
    print("=" * 40)
    
    # Configurar Kaggle
    if setup_kaggle():
        # Testar conexão
        print("\nTestando conexão...")
        if test_kaggle_connection():
            print("\n✅ Configuração concluída com sucesso!")
            print("Agora você pode usar o download automático de datasets.")
        else:
            print("\n❌ Erro na conexão. Verifique suas credenciais.")
    else:
        print("\n❌ Configuração não foi concluída.")


if __name__ == "__main__":
    main()
