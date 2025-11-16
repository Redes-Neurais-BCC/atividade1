"""
Script para executar as aplicações Streamlit
============================================
Permite escolher qual aplicação executar.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("="*80)
    print("APLICAÇÕES STREAMLIT DISPONÍVEIS")
    print("="*80)
    print("\n1. MLP Manual - Modelo MLP com configuração manual")
    print("   - Configure manualmente camadas, neurônios, learning rate")
    print("   - Visualizações avançadas integradas")
    print("   - Arquivo: src/interface/mlp_app.py")
    print()
    print("2. Otimização Avançada - Otimização automática com Optuna")
    print("   - Compara múltiplos modelos (Dense, LSTM, GRU, RNN)")
    print("   - Otimização automática de hiperparâmetros")
    print("   - Visualizações avançadas integradas")
    print("   - Arquivo: src/interface/advanced_app.py")
    print()
    print("3. Dashboard Completo - Interface completa com múltiplas funcionalidades")
    print("   - Arquivo: src/interface/streamlit_app.py")
    print()
    print("="*80)

    choice = input("\nEscolha a aplicação (1, 2 ou 3): ").strip()

    apps = {
        '1': 'src/interface/mlp_app.py',
        '2': 'src/interface/advanced_app.py',
        '3': 'src/interface/streamlit_app.py'
    }

    ports = {
        '1': 8501,
        '2': 8502,
        '3': 8503
    }

    if choice not in apps:
        print("❌ Escolha inválida!")
        return

    app_path = apps[choice]
    port = ports[choice]

    print(f"\n{'='*80}")
    print(f"INICIANDO APLICAÇÃO: {app_path}")
    print(f"Porta: {port}")
    print(f"URL: http://localhost:{port}")
    print(f"{'='*80}\n")
    print("Pressione Ctrl+C para parar o servidor\n")

    # Executar streamlit
    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        app_path,
        '--server.port',
        str(port),
        '--server.headless',
        'false'
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n✅ Aplicação encerrada.")


if __name__ == "__main__":
    main()
