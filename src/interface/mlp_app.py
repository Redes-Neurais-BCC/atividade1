"""
MLP Streamlit Application
=========================
Interface Streamlit para treinamento e avaliação de MLP
para previsão de resultados (Vitória/Derrota) de jogos do Dallas Mavericks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mlp_trainer import MLPTrainer
from models.feature_selector import FeatureSelector
import matplotlib.pyplot as plt
import io
from PIL import Image


# Configuração da página
st.set_page_config(
    page_title="MLP - Previsão de Resultados",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #0c5460;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# Header
st.markdown('<div class="main-header">🏀 MLP - Previsão de Resultados NBA</div>', unsafe_allow_html=True)
st.markdown("### Multi-Layer Perceptron com Backpropagation e Gradient Descent")
st.markdown("---")


# Sidebar - Configurações
st.sidebar.header("⚙️ Configurações do Modelo")

# Caminho do arquivo
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "dallas_games_2024-25.csv"

# Seleção de Features
st.sidebar.subheader("📊 Seleção de Features")
feature_method = st.sidebar.radio(
    "Método de Seleção:",
    options=['correlation', 'predefined', 'all'],
    format_func=lambda x: {
        'correlation': '🎯 Por Correlação (Top N)',
        'predefined': '📋 Predefinidas (Regressão Logística)',
        'all': '📦 Todas as Features'
    }[x],
    help="Escolha como selecionar as features para o modelo"
)

n_features = None
if feature_method == 'correlation':
    n_features = st.sidebar.slider(
        "Número de Features:",
        min_value=3,
        max_value=15,
        value=6,
        help="Seleciona as N features com maior correlação absoluta"
    )

# Arquitetura da Rede
st.sidebar.subheader("🧠 Arquitetura da Rede")
num_hidden_layers = st.sidebar.selectbox(
    "Número de Camadas Ocultas:",
    options=[1, 2, 3],
    index=1,
    help="Número de camadas ocultas na rede neural"
)

hidden_layers = []
for i in range(num_hidden_layers):
    neurons = st.sidebar.slider(
        f"Neurônios na Camada {i+1}:",
        min_value=3,
        max_value=50,
        value=[10, 5, 3][i] if i < 3 else 5,
        help=f"Número de neurônios na camada oculta {i+1}"
    )
    hidden_layers.append(neurons)

# Hiperparâmetros
st.sidebar.subheader("🎛️ Hiperparâmetros")
learning_rate = st.sidebar.select_slider(
    "Learning Rate:",
    options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    value=0.01,
    help="Taxa de aprendizado para o gradient descent"
)

epochs = st.sidebar.slider(
    "Épocas de Treinamento:",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Número de iterações de treinamento"
)

# Outras configurações
st.sidebar.subheader("⚡ Outras Configurações")
test_size = st.sidebar.slider(
    "Tamanho do Conjunto de Teste:",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
    help="Proporção dos dados reservados para teste"
)

random_state = st.sidebar.number_input(
    "Random Seed:",
    min_value=0,
    max_value=1000,
    value=42,
    help="Seed para reprodutibilidade dos resultados"
)

normalize_data = st.sidebar.checkbox(
    "Normalizar Dados",
    value=True,
    help="Aplica normalização (StandardScaler) nos dados"
)

# Botão de treinamento
st.sidebar.markdown("---")
train_button = st.sidebar.button(
    "🚀 Treinar Modelo",
    type="primary",
    use_container_width=True
)


# Função para criar gráficos
def plot_training_history(history_df):
    """Plota histórico de treinamento."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss durante o Treinamento', 'Acurácia durante o Treinamento')
    )

    # Loss
    fig.add_trace(
        go.Scatter(
            x=history_df['epoch'],
            y=history_df['loss'],
            mode='lines',
            name='Treino',
            line=dict(color='#0066cc', width=2)
        ),
        row=1, col=1
    )

    if 'val_loss' in history_df.columns and not history_df['val_loss'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=history_df['epoch'],
                y=history_df['val_loss'],
                mode='lines',
                name='Validação',
                line=dict(color='#ff6600', width=2)
            ),
            row=1, col=1
        )

    # Accuracy
    fig.add_trace(
        go.Scatter(
            x=history_df['epoch'],
            y=history_df['accuracy'],
            mode='lines',
            name='Treino',
            line=dict(color='#0066cc', width=2),
            showlegend=False
        ),
        row=1, col=2
    )

    if 'val_accuracy' in history_df.columns and not history_df['val_accuracy'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=history_df['epoch'],
                y=history_df['val_accuracy'],
                mode='lines',
                name='Validação',
                line=dict(color='#ff6600', width=2),
                showlegend=False
            ),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Época", row=1, col=1)
    fig.update_xaxes(title_text="Época", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Acurácia", row=1, col=2)

    fig.update_layout(
        height=400,
        showlegend=True,
        hovermode='x unified'
    )

    return fig


def plot_confusion_matrix(cm):
    """Plota matriz de confusão."""
    labels = ['Derrota', 'Vitória']

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='Blues',
        showscale=False
    ))

    fig.update_layout(
        title='Matriz de Confusão',
        xaxis_title='Predito',
        yaxis_title='Real',
        height=400,
        width=400
    )

    return fig


# Main content
if train_button:
    with st.spinner('🔄 Carregando dados e preparando modelo...'):
        # Inicializar trainer
        trainer = MLPTrainer(
            data_path=str(DATA_PATH),
            target_column='resultado',
            test_size=test_size,
            random_state=random_state
        )

        # Carregar dados
        df = trainer.load_data()

        # Mostrar informações dos dados
        st.markdown('<div class="sub-header">📊 Informações dos Dados</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Jogos", df.shape[0])
        with col2:
            victories = df['resultado'].sum()
            st.metric("Vitórias", victories)
        with col3:
            defeats = len(df) - victories
            st.metric("Derrotas", defeats)
        with col4:
            win_rate = (victories / len(df)) * 100
            st.metric("Taxa de Vitória", f"{win_rate:.1f}%")

        # Selecionar features
        if feature_method == 'correlation':
            features = trainer.select_features(method='correlation', n_features=n_features)
        elif feature_method == 'predefined':
            features = trainer.select_features(method='predefined')
        else:
            features = trainer.select_features(method='all')

        # Mostrar features selecionadas
        st.markdown('<div class="sub-header">🎯 Features Selecionadas</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box"><b>Método:</b> {feature_method}<br><b>Features ({len(features)}):</b> {", ".join(features)}</div>', unsafe_allow_html=True)

        # Mostrar correlações
        if feature_method == 'correlation':
            st.markdown("#### Correlações com o Target (resultado)")
            corr_data = []
            for feat in features:
                corr_val = trainer.feature_selector.correlation_scores[feat]
                corr_data.append({'Feature': feat, 'Correlação (abs)': corr_val})

            corr_df = pd.DataFrame(corr_data)

            fig_corr = px.bar(
                corr_df,
                x='Correlação (abs)',
                y='Feature',
                orientation='h',
                title='Correlação Absoluta das Features',
                color='Correlação (abs)',
                color_continuous_scale='Blues'
            )
            fig_corr.update_layout(height=300)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Preparar dados
        trainer.prepare_data(features, normalize=normalize_data)

        # Construir modelo
        trainer.build_model(
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            epochs=epochs
        )

        # Mostrar arquitetura
        st.markdown('<div class="sub-header">🧠 Arquitetura da Rede Neural</div>', unsafe_allow_html=True)
        arch = trainer.mlp.layer_sizes
        params = trainer.mlp.get_params()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <b>Camadas:</b> {arch}<br>
                <b>Input:</b> {arch[0]} features<br>
                <b>Camadas Ocultas:</b> {arch[1:-1]}<br>
                <b>Output:</b> {arch[-1]} (Vitória/Derrota)
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <b>Total de Parâmetros:</b> {params['num_parameters']}<br>
                <b>Learning Rate:</b> {learning_rate}<br>
                <b>Épocas:</b> {epochs}<br>
                <b>Normalização:</b> {'Sim' if normalize_data else 'Não'}
            </div>
            """, unsafe_allow_html=True)

    # Treinamento
    st.markdown('<div class="sub-header">🚀 Treinamento</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner('🔥 Treinando modelo...'):
        # Treinar
        trainer.train(verbose=False)
        progress_bar.progress(100)
        status_text.success(f'✅ Treinamento concluído em {trainer.training_time:.2f}s!')

    # Avaliação
    st.markdown('<div class="sub-header">📈 Resultados da Avaliação</div>', unsafe_allow_html=True)

    metrics = trainer.evaluate()

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "🎯 Acurácia",
            f"{metrics['accuracy']:.2%}",
            help="Proporção de predições corretas"
        )
    with col2:
        st.metric(
            "⚡ Precisão",
            f"{metrics['precision']:.2%}",
            help="Proporção de vitórias preditas que foram corretas"
        )
    with col3:
        st.metric(
            "📊 Recall",
            f"{metrics['recall']:.2%}",
            help="Proporção de vitórias reais que foram detectadas"
        )
    with col4:
        st.metric(
            "🎪 F1-Score",
            f"{metrics['f1_score']:.2%}",
            help="Média harmônica entre precisão e recall"
        )

    # Gráficos
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Evolução do Treinamento")
        history_df = trainer.get_training_history()
        fig_history = plot_training_history(history_df)
        st.plotly_chart(fig_history, use_container_width=True)

    with col2:
        st.markdown("#### Matriz de Confusão")
        fig_cm = plot_confusion_matrix(metrics['confusion_matrix'])
        st.plotly_chart(fig_cm, use_container_width=True)

    # Salvar modelo no session_state
    st.session_state['trainer'] = trainer
    st.session_state['features'] = features

    st.success("✅ Modelo treinado e avaliado com sucesso!")

    # ==================================================
    # VISUALIZACOES AVANCADAS
    # ==================================================
    st.markdown("---")
    st.markdown('<div class="sub-header">📊 Visualizações Avançadas</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Evolução do Erro", "📉 Matriz de Erros", "📊 Intervalos de Confiança"])

    with tab1:
        st.markdown("### Evolução do Erro Durante Treinamento")
        st.info("Mostra a evolução do erro (loss) e acurácia durante o treinamento. Detecta automaticamente overfitting e convergência.")

        try:
            # Gerar gráfico usando matplotlib
            fig_evolution = trainer.plot_training_evolution(show=False)
            st.pyplot(fig_evolution)
            plt.close(fig_evolution)
        except Exception as e:
            st.error(f"Erro ao gerar gráfico de evolução: {str(e)}")

    with tab2:
        st.markdown("### Matriz de Erros (3 Gráficos)")
        st.info("Análise detalhada dos erros: Histograma, Scatter Previsto vs Real, e Análise de Resíduos.")

        try:
            fig_errors = trainer.plot_error_matrix(show=False)
            st.pyplot(fig_errors)
            plt.close(fig_errors)
        except Exception as e:
            st.error(f"Erro ao gerar matriz de erros: {str(e)}")

    with tab3:
        st.markdown("### Intervalos de Confiança (Bootstrap)")
        st.info("Intervalos de confiança das previsões calculados usando método Bootstrap (1000 amostras).")

        try:
            fig_ci, intervals = trainer.plot_confidence_intervals(show=False)
            st.pyplot(fig_ci)
            plt.close(fig_ci)

            # Mostrar estatísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Média das Previsões", f"{intervals['mean']:.4f}")
            with col2:
                st.metric("IC 95% (Inferior)", f"{intervals['ci_lower']:.4f}")
            with col3:
                st.metric("IC 95% (Superior)", f"{intervals['ci_upper']:.4f}")

            st.metric("Desvio Padrão", f"{intervals['std']:.4f}")
        except Exception as e:
            st.error(f"Erro ao gerar intervalos de confiança: {str(e)}")


# Seção de predição
if 'trainer' in st.session_state:
    st.markdown("---")
    st.markdown('<div class="sub-header">🔮 Fazer Predição para Novo Jogo</div>', unsafe_allow_html=True)

    trainer = st.session_state['trainer']
    features = st.session_state['features']

    st.info("💡 **Dica:** Insira as estatísticas previstas para o próximo jogo e veja a predição do modelo!")

    # Criar inputs para features
    cols = st.columns(3)
    game_features = {}

    for i, feat in enumerate(features):
        with cols[i % 3]:
            # Obter estatísticas do dataset para valores padrão
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

    if st.button("🎯 Fazer Predição", type="primary"):
        result = trainer.predict_new_game(game_features)

        # Mostrar resultado
        col1, col2, col3 = st.columns(3)

        with col1:
            if result['prediction'] == 1:
                st.success(f"### ✅ {result['prediction_label']}")
            else:
                st.error(f"### ❌ {result['prediction_label']}")

        with col2:
            st.metric(
                "Probabilidade",
                f"{result['probability']:.2%}",
                help="Probabilidade da classe predita (Vitória)"
            )

        with col3:
            st.metric(
                "Confiança",
                f"{result['confidence']:.2%}",
                help="Confiança da predição"
            )

        # Gauge chart para probabilidade
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['probability'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilidade de Vitória (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkgreen" if result['prediction'] == 1 else "darkred"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)


# Informações sobre o modelo
with st.expander("ℹ️ Sobre o Modelo MLP"):
    st.markdown("""
    ### Multi-Layer Perceptron (MLP)

    Esta implementação utiliza uma **rede neural artificial totalmente conectada** (fully-connected)
    treinada do zero com **Backpropagation** e **Gradient Descent**.

    #### Características:
    - **Função de Ativação:** Sigmoid
    - **Loss Function:** Binary Cross-Entropy
    - **Otimizador:** Gradient Descent (batch)
    - **Inicialização:** Xavier/Glorot

    #### Arquitetura:
    - **Camada de Entrada:** Número de features selecionadas
    - **Camadas Ocultas:** Configuráveis (1-3 camadas)
    - **Camada de Saída:** 1 neurônio (classificação binária: Vitória/Derrota)

    #### Processo de Treinamento:
    1. **Forward Propagation:** Calcula as ativações através da rede
    2. **Loss Calculation:** Calcula o erro usando Binary Cross-Entropy
    3. **Backpropagation:** Calcula gradientes camada por camada
    4. **Gradient Descent:** Atualiza pesos e vieses
    5. **Repete** por N épocas

    #### Métodos de Seleção de Features:
    - **Por Correlação:** Seleciona as N features com maior correlação absoluta com o target
    - **Predefinidas:** Usa features dos modelos de regressão logística
    - **Todas:** Utiliza todas as features disponíveis no dataset
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🏀 MLP para Previsão de Resultados NBA | "
    "Desenvolvido com ❤️ usando Streamlit"
    "</div>",
    unsafe_allow_html=True
)
