"""
Advanced Neural Network Optimization App
=========================================
Interface Streamlit para otimizacao automatica de hiperparametros com Optuna.
Compara modelos Dense, RNN, LSTM e GRU.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Adicionar diretorio src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.advanced_trainer import AdvancedNeuralTrainer
import matplotlib.pyplot as plt


# Configuracao da pagina
st.set_page_config(
    page_title="Otimizacao Avancada - Neural Networks",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


# Header
st.markdown('<div class="main-header">🧠 Otimizacao Avancada de Redes Neurais</div>', unsafe_allow_html=True)
st.markdown("### Otimizacao Automatica com Optuna | Dense • RNN • LSTM • GRU")
st.markdown("---")


# Sidebar - Configuracoes
st.sidebar.header("⚙️ Configuracoes")

# Caminho do arquivo
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "dallas_games_2024-25.csv"

# Sempre usar otimização automática
tab_mode = "Otimizacao Automatica"

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Configuracoes de Otimizacao")

# Selecao de modelos para otimizar
st.sidebar.markdown("**Modelos a Otimizar:**")
optimize_dense = st.sidebar.checkbox("Dense (MLP)", value=True)
optimize_lstm = st.sidebar.checkbox("LSTM", value=True)
optimize_gru = st.sidebar.checkbox("GRU", value=True)
optimize_rnn = st.sidebar.checkbox("RNN (SimpleRNN)", value=False)

models_to_optimize = []
if optimize_dense:
    models_to_optimize.append('dense')
if optimize_lstm:
    models_to_optimize.append('lstm')
if optimize_gru:
    models_to_optimize.append('gru')
if optimize_rnn:
    models_to_optimize.append('rnn')

# Numero de trials
n_trials = st.sidebar.slider(
    "Total de Trials:",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
    help="Numero TOTAL de trials (Optuna distribuira automaticamente entre os modelos selecionados)"
)

# Timeout
use_timeout = st.sidebar.checkbox("Usar Timeout", value=False)
timeout = None
if use_timeout:
    timeout = st.sidebar.number_input(
        "Timeout (segundos):",
        min_value=60,
        max_value=3600,
        value=300,
        step=60
    )

# Selecao de Features
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Selecao de Features")
feature_method = st.sidebar.radio(
    "Metodo:",
    options=['correlation', 'predefined', 'all'],
    format_func=lambda x: {
        'correlation': '🎯 Por Correlacao',
        'predefined': '📋 Predefinidas',
        'all': '📦 Todas'
    }[x]
)

n_features = None
if feature_method == 'correlation':
    n_features = st.sidebar.slider(
        "Numero de Features:",
        min_value=3,
        max_value=15,
        value=8,
        help="Top N features por correlacao"
    )

# Configuracoes de dados
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Configuracoes de Dados")
test_size = st.sidebar.slider(
    "% Teste:",
    min_value=10,
    max_value=40,
    value=20,
    step=5
) / 100

val_size = st.sidebar.slider(
    "% Validacao (do treino):",
    min_value=10,
    max_value=30,
    value=15,
    step=5
) / 100

random_state = st.sidebar.number_input(
    "Random Seed:",
    min_value=0,
    max_value=1000,
    value=42
)

# Botao de inicio
st.sidebar.markdown("---")
start_button = st.sidebar.button(
    "🚀 Iniciar Otimizacao",
    type="primary",
    use_container_width=True,
    disabled=len(models_to_optimize) == 0
)

if len(models_to_optimize) == 0:
    st.sidebar.warning("⚠️ Selecione pelo menos um modelo")


# Main content - Apenas Otimizacao Automatica

# Informacao sobre Optuna
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
**🔬 Sobre a Otimizacao Automatica:**

Este aplicativo utiliza o **Optuna**, uma biblioteca de otimizacao de hiperparametros de ultima geracao,
para encontrar automaticamente a melhor configuracao de cada modelo.

**Processo:**
1. O Optuna testa diferentes combinacoes de hiperparametros E tipos de modelo
2. Os trials sao distribuidos automaticamente entre os modelos selecionados
3. Cada combinacao e avaliada no conjunto de validacao
4. O algoritmo TPE (Tree-structured Parzen Estimator) aprende com os trials anteriores
5. Ao final, o melhor modelo e treinado novamente no conjunto completo

**Hiperparametros Otimizados:**
- Numero de camadas e neuronios
- Funcao de ativacao
- Taxa de dropout
- Regularizacao L2
- Learning rate
- Otimizador (Adam, RMSprop)
- Batch size

**💡 Dica:** Para configuração manual, use o aplicativo MLP: `streamlit run src/interface/mlp_app.py`
""")
st.markdown('</div>', unsafe_allow_html=True)

if start_button:
    # Criar trainer
    with st.spinner('🔄 Inicializando...'):
        trainer = AdvancedNeuralTrainer(
            data_path=str(DATA_PATH),
            target_column='resultado',
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )

        # Carregar dados
        df = trainer.load_data()

    # Mostrar info dos dados
    st.markdown('<div class="sub-header">📊 Informacoes dos Dados</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Jogos", df.shape[0])
    with col2:
        victories = df['resultado'].sum()
        st.metric("Vitorias", victories)
    with col3:
        defeats = len(df) - victories
        st.metric("Derrotas", defeats)
    with col4:
        win_rate = (victories / len(df)) * 100
        st.metric("Taxa de Vitoria", f"{win_rate:.1f}%")

    # Selecionar features
    with st.spinner('🎯 Selecionando features...'):
        if feature_method == 'correlation':
            features = trainer.select_features(method='correlation', n_features=n_features)
        elif feature_method == 'predefined':
            features = trainer.select_features(method='predefined')
        else:
            features = trainer.select_features(method='all')

    st.success(f"✅ {len(features)} features selecionadas: {', '.join(features)}")

    # Preparar dados
    with st.spinner('⚙️ Preparando dados...'):
        trainer.prepare_data(features, normalize=True)

    # Otimizacao
    st.markdown('<div class="sub-header">🔍 Otimizacao de Hiperparametros</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warning-box">
    <b>⏳ Iniciando otimizacao...</b><br>
    Modelos a testar: {', '.join([m.upper() for m in models_to_optimize])}<br>
    Total de trials: {n_trials}<br>
    (Optuna distribuira automaticamente os trials entre os {len(models_to_optimize)} modelos)<br><br>
    Este processo pode levar alguns minutos. Por favor, aguarde...
    </div>
    """, unsafe_allow_html=True)

    # Progress container
    progress_container = st.empty()
    status_container = st.empty()

    # Executar otimizacao
    try:
        optimization_results = trainer.optimize_hyperparameters(
            model_types=models_to_optimize,
            n_trials=n_trials,
            timeout=timeout
        )

        # Resultados da otimizacao
        st.markdown('<div class="sub-header">📈 Resultados da Otimizacao</div>', unsafe_allow_html=True)

        # Distribuicao de trials
        trials_per_model = optimization_results.get('trials_per_model', {})

        if trials_per_model:
            st.markdown("### Distribuicao de Trials por Modelo:")

            trials_data = pd.DataFrame([
                {'Modelo': model.upper(), 'Trials Completados': count}
                for model, count in sorted(trials_per_model.items())
            ])

            fig_trials = px.bar(
                trials_data,
                x='Modelo',
                y='Trials Completados',
                title=f'Distribuicao dos {optimization_results["n_trials_total"]} Trials',
                color='Trials Completados',
                color_continuous_scale='Blues',
                text='Trials Completados'
            )
            fig_trials.update_traces(textposition='outside')
            fig_trials.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_trials, use_container_width=True)

            st.info(f"💡 Optuna distribuiu automaticamente os {optimization_results['n_trials_total']} trials "
                   f"entre os {len(trials_per_model)} modelos, focando nos mais promissores.")

        # Melhor modelo
        best_model_type = optimization_results['best_model_type']
        best_score = optimization_results['best_score']

        st.markdown(f"""
        <div class="success-box">
        <h3>🏆 Melhor Modelo Encontrado: {best_model_type.upper()}</h3>
        <p><b>Acuracia de Validacao:</b> {best_score:.2%}</p>
        <p><b>Trials Testados:</b> {trials_per_model.get(best_model_type, 0)} para este modelo</p>
        <p><b>Tempo Total de Otimizacao:</b> {optimization_results['total_time']:.2f}s</p>
        </div>
        """, unsafe_allow_html=True)

        # Melhores hiperparametros
        st.markdown("### Melhores Hiperparametros Encontrados:")
        best_params = trainer.optimizer.best_params

        params_df = pd.DataFrame([
            {'Parametro': k, 'Valor': v}
            for k, v in best_params.items()
        ])
        st.dataframe(params_df, use_container_width=True)

        # Treinar modelo final
        st.markdown('<div class="sub-header">🚀 Treinamento do Modelo Final</div>', unsafe_allow_html=True)

        with st.spinner('🔥 Treinando modelo final com os melhores hiperparametros...'):
            model, final_metrics = trainer.train_best_model(epochs=300, verbose=0)

        # Metricas finais
        st.markdown("### Metricas no Conjunto de Teste:")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🎯 Acuracia", f"{final_metrics['accuracy']:.2%}")
        with col2:
            st.metric("⚡ Precisao", f"{final_metrics['precision']:.2%}")
        with col3:
            st.metric("📊 Recall", f"{final_metrics['recall']:.2%}")
        with col4:
            st.metric("🎪 F1-Score", f"{final_metrics['f1_score']:.2%}")
        with col5:
            st.metric("📈 R2-Score", f"{final_metrics['r2_score']:.3f}")

        # Historico de treinamento
        st.markdown("### Evolucao do Treinamento:")

        history = final_metrics['history']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Loss', 'Acuracia')
        )

        # Loss
        fig.add_trace(
            go.Scatter(
                y=history['loss'],
                mode='lines',
                name='Treino',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                y=history['val_loss'],
                mode='lines',
                name='Validacao',
                line=dict(color='#ff7f0e', width=2)
            ),
            row=1, col=1
        )

        # Accuracy
        fig.add_trace(
            go.Scatter(
                y=history['accuracy'],
                mode='lines',
                name='Treino',
                line=dict(color='#1f77b4', width=2),
                showlegend=False
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                y=history['val_accuracy'],
                mode='lines',
                name='Validacao',
                line=dict(color='#ff7f0e', width=2),
                showlegend=False
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Epoca", row=1, col=1)
        fig.update_xaxes(title_text="Epoca", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Acuracia", row=1, col=2)
        fig.update_layout(height=400, hovermode='x unified')

        st.plotly_chart(fig, use_container_width=True)

        # Informacoes do modelo
        model_summary = final_metrics['model_summary']

        st.markdown("### Arquitetura do Modelo:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Nome:** {model_summary['name']}
            **Tipo:** {best_model_type.upper()}
            **Camadas:** {model_summary['layers']}
            """)
        with col2:
            st.markdown(f"""
            **Parametros Totais:** {model_summary['total_params']:,}
            **Parametros Treinaveis:** {model_summary['trainable_params']:,}
            **Formato de Entrada:** {model_summary['input_shape']}
            """)

        # Salvar no session_state
        st.session_state['advanced_trainer'] = trainer
        st.session_state['advanced_features'] = features
        st.session_state['best_model_type'] = best_model_type

        st.success("✅ Otimizacao e treinamento concluidos com sucesso!")

        # ==================================================
        # VISUALIZACOES AVANCADAS
        # ==================================================
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Visualizacoes Avancadas</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Evolucao do Erro", "📉 Matriz de Erros", "📊 Intervalos de Confianca"])

        with tab1:
            st.markdown("### Evolucao do Erro Durante Treinamento")
            st.info("Mostra a evolucao do erro (loss) e acuracia durante o treinamento. Detecta automaticamente overfitting e convergencia.")

            try:
                fig_evolution = trainer.plot_training_evolution(show=False)
                st.pyplot(fig_evolution)
                plt.close(fig_evolution)
            except Exception as e:
                st.error(f"Erro ao gerar grafico de evolucao: {str(e)}")

        with tab2:
            st.markdown("### Matriz de Erros (3 Graficos)")
            st.info("Analise detalhada dos erros: Histograma, Scatter Previsto vs Real, e Analise de Residuos.")

            try:
                fig_errors = trainer.plot_error_matrix(show=False)
                st.pyplot(fig_errors)
                plt.close(fig_errors)
            except Exception as e:
                st.error(f"Erro ao gerar matriz de erros: {str(e)}")

        with tab3:
            st.markdown("### Intervalos de Confianca (Bootstrap)")
            st.info("Intervalos de confianca das previsoes calculados usando metodo Bootstrap (1000 amostras).")

            try:
                fig_ci, intervals = trainer.plot_confidence_intervals(show=False)
                st.pyplot(fig_ci)
                plt.close(fig_ci)

                # Mostrar estatisticas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Media das Previsoes", f"{intervals['mean']:.4f}")
                with col2:
                    st.metric("IC 95% (Inferior)", f"{intervals['ci_lower']:.4f}")
                with col3:
                    st.metric("IC 95% (Superior)", f"{intervals['ci_upper']:.4f}")

                st.metric("Desvio Padrao", f"{intervals['std']:.4f}")
            except Exception as e:
                st.error(f"Erro ao gerar intervalos de confianca: {str(e)}")

    except Exception as e:
        st.error(f"❌ Erro durante a otimizacao: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# Secao de predicao
if 'advanced_trainer' in st.session_state:
    st.markdown("---")
    st.markdown('<div class="sub-header">🔮 Fazer Predicao</div>', unsafe_allow_html=True)

    trainer = st.session_state['advanced_trainer']
    features = st.session_state['advanced_features']

    st.info(f"💡 Usando modelo: **{st.session_state['best_model_type'].upper()}**")

    # Inputs
    cols = st.columns(3)
    game_features = {}

    for i, feat in enumerate(features):
        with cols[i % 3]:
            mean_val = trainer.df[feat].mean()
            min_val = trainer.df[feat].min()
            max_val = trainer.df[feat].max()

            game_features[feat] = st.number_input(
                feat.replace('-', ' ').title(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=0.01 if trainer.df[feat].dtype == 'float64' else 1.0,
                format="%.3f" if trainer.df[feat].dtype == 'float64' else "%.0f"
            )

    if st.button("🎯 Prever Resultado", type="primary"):
        result = trainer.predict_new_game(game_features)

        col1, col2, col3 = st.columns(3)

        with col1:
            if result['prediction'] == 1:
                st.success(f"### ✅ {result['prediction_label']}")
            else:
                st.error(f"### ❌ {result['prediction_label']}")

        with col2:
            st.metric("Probabilidade", f"{result['probability']:.2%}")

        with col3:
            st.metric("Confianca", f"{result['confidence']:.2%}")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🧠 Otimizacao Avancada com Optuna | "
    "Desenvolvido com ❤️ usando TensorFlow, Optuna e Streamlit"
    "</div>",
    unsafe_allow_html=True
)
