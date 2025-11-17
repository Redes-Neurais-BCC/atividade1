import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="Dallas Mavericks 2024-25 - Análise Exploratória",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        border: 1px solid #d1ecf1;
    }
    .stMetric {
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border: 1px solid #gray !important;
        margin: 0.25rem !important;
    }
    .stMetric > div {
        background-color: #gray !important;
    }
    .stMetric [data-testid="metric-container"] {
        background-color: #e8f4fd !important;
        border: 1px solid #bee5eb !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        ROOT_DIR = Path(__file__).resolve().parents[2]
        processed_dir = ROOT_DIR / "data" / "processed"
        original_dir = ROOT_DIR / "data" / "original"
        
        players_df = pd.read_csv(processed_dir / "dallas_players_2024-25.csv")
        games_df = pd.read_csv(processed_dir / "dallas_games_2024-25.csv")
        
        original_players_df = pd.read_csv(original_dir / "dal_players_season_stats_media_2024_25.csv")
        
        players_df = add_player_names(players_df, original_players_df)
        
        games_df['data-jogo'] = pd.to_datetime(games_df['data-jogo'], format='%Y%m%d')
        
        return players_df, games_df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

def add_player_names(processed_df, original_df):
    """Verifica e adiciona os nomes dos jogadores aos dados processados"""
    try:
        if 'nome-jogador' in processed_df.columns:
            valid_names = processed_df['nome-jogador'].notna() & (processed_df['nome-jogador'] != "")

            if valid_names.sum() > len(processed_df) * 0.5:
                return processed_df

        processed_with_names = processed_df.copy()

        if 'nome-jogador' not in processed_with_names.columns:
            processed_with_names['nome-jogador'] = ""

        for idx, row in processed_df.iterrows():
            if pd.notna(row.get('nome-jogador')) and row.get('nome-jogador', '').strip() != "":
                continue

            matching_player = original_df[
                (abs(original_df['AGE'] - row['idade']) <= 1) &
                (abs(original_df['GP'] - row['jogos-disputados_total']) <= 2) &
                (abs(original_df['MIN'] - row['minutos_media']) <= 2.0) &
                (abs(original_df['PTS'] - row['pontos_media']) <= 1.0)
            ]

            if len(matching_player) >= 1:
                processed_with_names.at[idx, 'nome-jogador'] = matching_player.iloc[0]['PLAYER_NAME']
            else:
                matching_player = original_df[
                    (abs(original_df['AGE'] - row['idade']) <= 2) &
                    (abs(original_df['GP'] - row['jogos-disputados_total']) <= 5)
                ]

                if len(matching_player) >= 1:
                    processed_with_names.at[idx, 'nome-jogador'] = matching_player.iloc[0]['PLAYER_NAME']
                else:
                    position_name = {1: 'Guard', 2: 'Forward', 3: 'Forward-Center', 4: 'Center-Forward', 5: 'Center'}
                    pos = position_name.get(row.get('posicao-g-f-fc-cf-c', 0), 'Player')
                    processed_with_names.at[idx, 'nome-jogador'] = f"{pos} #{idx+1}"

        return processed_with_names
    except Exception as e:
        st.warning(f"Não foi possível processar nomes dos jogadores: {e}")
        if 'nome-jogador' not in processed_df.columns:
            processed_df['nome-jogador'] = [f"Jogador #{i+1}" for i in range(len(processed_df))]
        return processed_df

def create_summary_metrics(players_df, games_df):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_players = len(players_df)
        st.metric("Total de Jogadores", total_players)
    
    with col2:
        total_games = len(games_df)
        st.metric("Jogos Disputados", total_games)
    
    with col3:
        wins = games_df['resultado'].sum()
        win_pct = (wins / total_games * 100) if total_games > 0 else 0
        st.metric("Vitórias", f"{wins} ({win_pct:.1f}%)")
    
    with col4:
        avg_points = games_df['pontos'].mean()
        st.metric("Média de Pontos", f"{avg_points:.1f}")
    
    with col5:
        avg_assists = games_df['assistencias'].mean()
        st.metric("Média de Assistências", f"{avg_assists:.1f}")

def player_analysis(players_df):
    st.header("📊 Análise dos Jogadores")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 10 Pontuadores")
        top_scorers = players_df.nlargest(10, 'pontos_total')[['nome-jogador', 'posicao-g-f-fc-cf-c', 'pontos_total', 'jogos-disputados_total']]
        top_scorers['pontos_por_jogo'] = top_scorers['pontos_total'] / top_scorers['jogos-disputados_total']

        top_scorers_chart = top_scorers.reset_index()

        fig = px.bar(
            top_scorers_chart,
            x='pontos_total',
            y='nome-jogador',
            orientation='h',
            title="Pontos por Jogador",
            labels={'pontos_total': 'Pontos Totais', 'nome-jogador': 'Jogadores'},
            color='pontos_total',
            color_continuous_scale='blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Eficiência de Arremessos")
        efficiency_df = players_df[players_df['arremessos-tentados_total'] >= 5].copy()
        efficiency_df['eficiencia_arremesso'] = efficiency_df['porcentagem-arremessos_media'] * 100

        fig = px.scatter(
            efficiency_df,
            x='arremessos-tentados_total',
            y='eficiencia_arremesso',
            size='pontos_total',
            color='porcentagem-triplos_media',
            hover_name='nome-jogador' if 'nome-jogador' in efficiency_df.columns else None,
            title="Eficiência vs Volume de Arremessos",
            labels={
                'arremessos-tentados_total': 'Arremessos Tentados',
                'eficiencia_arremesso': 'Eficiência (%)',
                'porcentagem-triplos_media': '% Triplos'
            },
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("👥 Distribuição por Posição")
    col1, col2 = st.columns(2)
    
    with col1:
        pos_counts = players_df['posicao-g-f-fc-cf-c'].value_counts()
        position_names = {1: '1 - Guard', 2: '2 - Forward', 3: '3 - Forward-Center', 4: '4 - Center-Forward', 5: '5 - Center'}
        
        ordered_positions = sorted([pos for pos in pos_counts.index if pos in position_names.keys()])
        ordered_values = [pos_counts[pos] for pos in ordered_positions]
        ordered_labels = [position_names[pos] for pos in ordered_positions]
        
        fig = px.pie(
            values=ordered_values,
            names=ordered_labels,
            title="Distribuição de Jogadores por Posição",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        pos_stats = players_df.groupby('posicao-g-f-fc-cf-c').agg({
            'pontos_media': 'mean',
            'rebotes-totais_media': 'mean',
            'assistencias_media': 'mean',
            'porcentagem-arremessos_media': 'mean'
        }).round(2)

        pos_stats = pos_stats.sort_index()
        position_labels = {1: '1 - Guard', 2: '2 - Forward', 3: '3 - Forward-Center', 4: '4 - Center-Forward', 5: '5 - Center'}
        pos_stats.index = [position_labels.get(pos, f"Posição {pos}") for pos in pos_stats.index]
        pos_stats.index.name = 'Posição'  

        st.write("**Médias por Posição:**")
        st.dataframe(pos_stats, use_container_width=True)

def game_analysis(games_df):
    """Análise detalhada dos jogos"""
    st.header("🏀 Análise dos Jogos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance ao Longo da Temporada")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=games_df['data-jogo'],
            y=games_df['pontos'],
            mode='lines+markers',
            name='Pontos',
            line=dict(color='blue', width=2),
            marker=dict(
                color=games_df['resultado'].map({1: 'green', 0: 'red'}),
                size=8,
                line=dict(color='white', width=1)
            )
        ))
        
        fig.update_layout(
            title="Pontos por Jogo (Verde=Vitória, Vermelho=Derrota)",
            xaxis_title="Data do Jogo",
            yaxis_title="Pontos",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏠 Performance Casa vs Fora")
        
        home_away = games_df.groupby('mando-de-jogo').agg({
            'pontos': 'mean',
            'porcentagem-arremessos': 'mean',
            'assistencias': 'mean',
            'resultado': 'mean'
        }).round(3)
        
        home_away.index = ['Fora de Casa', 'Em Casa']
        
        # Converter porcentagem de arremessos para percentual (0-100)
        home_away['porcentagem-arremessos'] = home_away['porcentagem-arremessos'] * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Pontos Médios',
            x=['Fora de Casa', 'Em Casa'],
            y=home_away['pontos'],
            marker_color='blue',
            text=[f"{val:.1f}" for val in home_away['pontos']],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='% Arremessos',
            x=['Fora de Casa', 'Em Casa'],
            y=home_away['porcentagem-arremessos'],
            marker_color='green',
            text=[f"{val:.1f}%" for val in home_away['porcentagem-arremessos']],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Assistências',
            x=['Fora de Casa', 'Em Casa'],
            y=home_away['assistencias'],
            marker_color='orange',
            text=[f"{val:.1f}" for val in home_away['assistencias']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Comparação Casa vs Fora",
            barmode='group',
            height=400,
            xaxis_title="Local do Jogo",
            yaxis_title="Valores",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔗 Correlações entre Estatísticas")
    
    numeric_cols = [
        'pontos', 'arremessos-convertidos', 'porcentagem-arremessos',
        'triplos-convertidos', 'porcentagem-triplos', 'rebotes-totais',
        'assistencias', 'roubos', 'tocos', 'resultado'
    ]
    
    correlation_matrix = games_df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Matriz de Correlação - Estatísticas dos Jogos",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def advanced_analysis(players_df, games_df):
    """Análises mais avançadas"""
    st.header("🧠 Análise Avançada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Eficiência vs Uso")

        players_df['uso_estimado'] = (
            players_df['arremessos-tentados_total'] +
            players_df['lances-livres-tentados_total'] * 0.44 +
            players_df['erros_total']
        ) / players_df['minutos_total']

        players_df['eficiencia_verdadeira'] = (
            players_df['pontos_total'] /
            (2 * (players_df['arremessos-tentados_total'] + 0.44 * players_df['lances-livres-tentados_total']))
        )

        regular_players = players_df[players_df['minutos_total'] >= 100]
        
        fig = px.scatter(
            regular_players,
            x='uso_estimado',
            y='eficiencia_verdadeira',
            size='minutos_total',
            color='pontos_total',
            hover_name='nome-jogador' if 'nome-jogador' in regular_players.columns else None,
            title="Eficiência Verdadeira vs Taxa de Uso",
            labels={
                'uso_estimado': 'Taxa de Uso Estimada',
                'eficiencia_verdadeira': 'Eficiência Verdadeira',
                'minutos_total': 'Minutos Totais',
                'pontos_total': 'Pontos Totais'
            },
            color_continuous_scale='plasma'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Análise de Clutch Time")
        
        games_df['jogo_apertado'] = abs(games_df['saldo-pontos']) <= 10
        clutch_performance = games_df.groupby('jogo_apertado').agg({
            'pontos': 'mean',
            'porcentagem-arremessos': 'mean',
            'erros': 'mean',
            'resultado': 'mean'
        }).round(3)
        
        clutch_performance.index = ['Jogos Folgados', 'Jogos Apertados']
        
        categories = ['Pontos Médios', '% Arremessos', 'Erros (inv)', '% Vitórias']
        
        fig = go.Figure()
        
        for idx, game_type in enumerate(clutch_performance.index):
            values = [
                clutch_performance.loc[game_type, 'pontos'],
                clutch_performance.loc[game_type, 'porcentagem-arremessos'] * 100,
                (1 - clutch_performance.loc[game_type, 'erros'] / 20) * 100, 
                clutch_performance.loc[game_type, 'resultado'] * 100
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=game_type
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 120]
                )
            ),
            showlegend=True,
            title="Performance: Jogos Apertados vs Folgados",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def interactive_analysis(players_df, games_df):
    """Análise interativa com filtros"""
    st.header("🔍 Análise Interativa")
    
    # Os filtros agora são globais na sidebar
    filtered_players = players_df  # Já vem filtrado da função main
    
    st.write("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stat_options = {
            'pontos_media': 'Pontos por Jogo',
            'rebotes-totais_media': 'Rebotes por Jogo',
            'assistencias_media': 'Assistências por Jogo',
            'porcentagem-arremessos_media': '% Arremessos',
            'porcentagem-triplos_media': '% Triplos',
            'roubos_media': 'Roubos por Jogo',
            'tocos_media': 'Tocos por Jogo'
        }

        x_stat = st.selectbox("Variável Independente X", options=list(stat_options.keys()),
                             format_func=lambda x: stat_options[x])

    with col2:
        y_stat = st.selectbox("Variável Dependente Y", options=list(stat_options.keys()),
                             format_func=lambda x: stat_options[x], index=1)
    
    if not filtered_players.empty:
        fig = px.scatter(
            filtered_players,
            x=x_stat,
            y=y_stat,
            size='minutos_total',
            color='posicao-g-f-fc-cf-c',
            hover_name='nome-jogador' if 'nome-jogador' in filtered_players.columns else None,
            title=f"{stat_options[y_stat]} vs {stat_options[x_stat]}",
            labels={x_stat: stat_options[x_stat], y_stat: stat_options[y_stat]},
            color_discrete_map={1: 'blue', 2: 'green', 3: 'red', 4: 'purple', 5: 'orange'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Jogadores Selecionados")
        display_cols = ['nome-jogador', 'posicao-g-f-fc-cf-c', 'idade', 'jogos-disputados_total', 'minutos_media',
                       'pontos_media', 'rebotes-totais_media', 'assistencias_media', 'porcentagem-arremessos_media']

        available_cols = [col for col in display_cols if col in filtered_players.columns]
        
        display_df = filtered_players[available_cols].copy().round(2)
        
        if 'posicao-g-f-fc-cf-c' in display_df.columns:
            position_labels = {1: '1 - Guard', 2: '2 - Forward', 3: '3 - Forward-Center', 4: '4 - Center-Forward', 5: '5 - Center'}
            display_df['posicao-g-f-fc-cf-c'] = display_df['posicao-g-f-fc-cf-c'].map(position_labels)
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Nenhum jogador atende aos critérios selecionados.")

def regression_analysis(players_df, games_df):
    """Análise de Regressão Linear e Logística"""
    st.header("📈 Análise Preditiva - Regressão Linear e Logística")
    
    if not SKLEARN_AVAILABLE:
        st.error("⚠️ Scikit-learn não está instalado. Instale com: pip install scikit-learn")
        return
    
    analysis_type = st.selectbox(
        "Tipo de Análise",
        ["Regressão Linear", "Regressão Logística"],
        help="Escolha o tipo de análise preditiva"
    )
    
    st.markdown("---")
    
    if analysis_type == "Regressão Linear":
        linear_regression_analysis(players_df, games_df)
    else:
        logistic_regression_analysis(players_df, games_df)

def linear_regression_analysis(players_df, games_df):
    """Análise de Regressão Linear"""
    st.subheader("🔢 Regressão Linear")
    st.markdown("**Predição de valores numéricos (pontos, rebotes, assistências)**")
    
    df = players_df.copy()
    
    df = df.dropna()
    
    if len(df) < 10:
        st.warning("Dados insuficientes para análise de regressão.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Configuração da Análise")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_options = {
            'pontos_media': 'Pontos por Jogo',
            'rebotes-totais_media': 'Rebotes por Jogo',
            'assistencias_media': 'Assistências por Jogo',
            'porcentagem-arremessos_media': 'Porcentagem de Arremessos',
            'minutos_media': 'Minutos por Jogo'
        }

        target_var = st.selectbox(
            "Variável Dependente (Y) - O que queremos prever:",
            options=[col for col in target_options.keys() if col in numeric_cols],
            format_func=lambda x: target_options.get(x, x),
            help="Esta é a variável que queremos prever"
        )

        feature_options = {
            'jogos-disputados_total': 'Jogos Disputados',
            'minutos_media': 'Minutos por Jogo',
            'arremessos-tentados_media': 'Arremessos Tentados',
            'arremessos-convertidos_media': 'Arremessos Convertidos',
            'porcentagem-arremessos_media': 'Porcentagem de Arremessos',
            'triplos-tentados_media': 'Triplos Tentados',
            'triplos-convertidos_media': 'Triplos Convertidos',
            'porcentagem-triplos_media': 'Porcentagem de Triplos',
            'lances-livres-tentados_media': 'Lances Livres Tentados',
            'lances-livres-convertidos_media': 'Lances Livres Convertidos',
            'rebotes-ofensivos_media': 'Rebotes Ofensivos',
            'rebotes-defensivos_media': 'Rebotes Defensivos',
            'idade': 'Idade',
            'altura-cm': 'Altura (cm)',
            'peso-kg': 'Peso (kg)'
        }
        
        available_features = [col for col in feature_options.keys() if col in numeric_cols and col != target_var]
        
        selected_features = st.multiselect(
            "Variáveis Independentes (X) - Fatores que influenciam:",
            options=available_features,
            default=available_features[:3] if len(available_features) >= 3 else available_features,
            format_func=lambda x: feature_options.get(x, x),
            help="Estas são as variáveis que podem influenciar nossa predição"
        )
        
        test_size = st.slider(
            "Porcentagem para teste (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Porcentagem dos dados reservada para testar o modelo"
        )
    
    with col2:
        st.subheader("📊 Informações dos Dados")
        
        if target_var and selected_features:
            X = df[selected_features]
            y = df[target_var]
            
            st.write("**Estatísticas da Variável Dependente:**")
            stats_df = pd.DataFrame({
                'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo'],
                'Valor': [y.mean(), y.median(), y.std(), y.min(), y.max()]
            })
            st.dataframe(stats_df.round(2), use_container_width=True)
            
            st.write(f"**Tamanho do dataset:** {len(df)} registros")
            st.write(f"**Variáveis independentes:** {len(selected_features)}")
    
    if target_var and selected_features and len(selected_features) > 0:
        st.markdown("---")
        
        X = df[selected_features]
        y = df[target_var]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Métricas do Modelo")
            st.metric("R² (Treino)", f"{train_r2:.3f}")
            st.metric("R² (Teste)", f"{test_r2:.3f}")
            st.metric("RMSE (Treino)", f"{train_rmse:.2f}")
            st.metric("RMSE (Teste)", f"{test_rmse:.2f}")
        
        with col2:
            st.subheader("⚙️ Coeficientes")
            coef_df = pd.DataFrame({
                'Variável': selected_features,
                'Coeficiente': model.coef_,
                'Impacto': ['Alto' if abs(c) > np.std(model.coef_) else 'Baixo' for c in model.coef_]
            })
            st.dataframe(coef_df.round(4), use_container_width=True)
            
            st.write(f"**Intercepto (β₀):** {model.intercept_:.4f}")
        
        with col3:
            st.subheader("🔮 Fazer Predição")
            st.write("Insira valores para fazer uma predição:")
            
            prediction_values = {}
            for feature in selected_features:
                mean_val = X[feature].mean()
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                
                prediction_values[feature] = st.number_input(
                    feature_options.get(feature, feature),
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"pred_{feature}"
                )
            
            if st.button("🎯 Fazer Predição"):
                pred_input = np.array([list(prediction_values.values())])
                prediction = model.predict(pred_input)[0]
                st.success(f"**Predição:** {prediction:.2f}")
        
        st.markdown("---")
        st.subheader("📊 Visualizações")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dispersão com Linha de Regressão",
            "🎯 Predição vs Realidade",
            "🔄 Matriz de Erro Categorizado",
            "📈 Análise de Erro por Faixa",
            "📉 Tendência com Intervalo de Confiança"
        ])
        
        with tab1:
            # Diagrama de Dispersão com Linha de Regressão
            if len(selected_features) > 0:
                first_feature = selected_features[0]
                first_feature_name = feature_options.get(first_feature, first_feature)
                target_name = target_options[target_var]

                fig = px.scatter(
                    df,
                    x=first_feature,
                    y=target_var,
                    title=f"Dispersão com Linha de Regressão: {target_name} vs {first_feature_name}",
                    labels={
                        first_feature: first_feature_name,
                        target_var: target_name
                    },
                    trendline="ols",
                    trendline_color_override="red"
                )

                fig.update_layout(
                    height=500,
                    xaxis_title=first_feature_name,
                    yaxis_title=target_name
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"📌 A linha vermelha representa a melhor linha reta que se ajusta aos dados (regressão linear).")

        with tab2:
            # Gráfico de Predição vs. Realidade
            pred_real_df = pd.DataFrame({
                'Real': np.concatenate([y_train, y_test]),
                'Predito': np.concatenate([y_pred_train, y_pred_test]),
                'Tipo': ['Treino'] * len(y_train) + ['Teste'] * len(y_test)
            })

            fig = px.scatter(
                pred_real_df,
                x='Real',
                y='Predito',
                color='Tipo',
                title="Valores Preditos vs Valores Reais",
                labels={
                    'Real': f'Valor Real ({target_options[target_var]})',
                    'Predito': f'Valor Predito ({target_options[target_var]})',
                    'Tipo': 'Conjunto de Dados'
                },
                color_discrete_map={'Treino': 'blue', 'Teste': 'red'}
            )

            min_val = min(pred_real_df['Real'].min(), pred_real_df['Predito'].min())
            max_val = max(pred_real_df['Real'].max(), pred_real_df['Predito'].max())

            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predição Perfeita',
                line=dict(dash='dash', color='green', width=2),
                hovertemplate='<b>Linha de Predição Perfeita</b><br>Real: %{x:.2f}<br>Predito: %{y:.2f}<extra></extra>'
            ))

            fig.update_layout(
                height=500,
                xaxis_title=f'Valor Real ({target_options[target_var]})',
                yaxis_title=f'Valor Predito ({target_options[target_var]})'
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"📌 Pontos próximos à linha verde (diagonal) indicam predições mais precisas.")

            # Métricas de erro
            col1, col2, col3 = st.columns(3)
            with col1:
                mae_test = np.mean(np.abs(y_test - y_pred_test))
                st.metric("MAE (Teste)", f"{mae_test:.2f}")
            with col2:
                st.metric("RMSE (Teste)", f"{test_rmse:.2f}")
            with col3:
                st.metric("R² (Teste)", f"{test_r2:.3f}")

        with tab3:
            # Matriz de Erro Categorizado (para Regressão Linear)
            target_name = target_options[target_var]

            # Categorizar valores em 3 faixas: Baixo, Médio, Alto
            y_all = np.concatenate([y_train, y_test])
            y_pred_all = np.concatenate([y_pred_train, y_pred_test])

            # Definir quartis para categorização
            q1, q2, q3 = np.percentile(y_all, [25, 50, 75])

            def categorize(values):
                categories = []
                for v in values:
                    if v <= q1:
                        categories.append('Baixo')
                    elif v <= q3:
                        categories.append('Médio')
                    else:
                        categories.append('Alto')
                return categories

            y_real_cat = categorize(y_test)
            y_pred_cat = categorize(y_pred_test)

            # Criar matriz de confusão para categorias
            from sklearn.metrics import confusion_matrix as cm_sklearn
            categories_order = ['Baixo', 'Médio', 'Alto']
            cm_cat = cm_sklearn(y_real_cat, y_pred_cat, labels=categories_order)

            # Criar heatmap
            fig = px.imshow(
                cm_cat,
                title=f"Matriz de Erro Categorizado - {target_name}",
                labels=dict(x="Categoria Predita", y="Categoria Real", color="Quantidade"),
                x=categories_order,
                y=categories_order,
                color_continuous_scale='Blues',
                text_auto=True
            )

            fig.update_layout(
                height=500,
                xaxis_title="Categoria Predita",
                yaxis_title="Categoria Real"
            )

            fig.update_traces(
                hovertemplate="<b>Real:</b> %{y}<br><b>Predito:</b> %{x}<br><b>Quantidade:</b> %{z}<extra></extra>"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"📌 Valores de {target_name} foram categorizados em 3 faixas: Baixo (≤{q1:.1f}), Médio ({q1:.1f}-{q3:.1f}), Alto (≥{q3:.1f}). Diagonal principal mostra predições corretas.")

            # Mostrar acurácia categórica
            correct_cat = sum([1 for r, p in zip(y_real_cat, y_pred_cat) if r == p])
            accuracy_cat = correct_cat / len(y_real_cat)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Acurácia Categórica", f"{accuracy_cat:.1%}")
            with col2:
                st.metric("Predições Corretas", f"{correct_cat}/{len(y_real_cat)}")

        with tab4:
            # Análise de Erro por Faixa de Valores
            target_name = target_options[target_var]

            # Criar faixas de valores
            y_test_array = np.array(y_test)
            y_pred_test_array = np.array(y_pred_test)

            # Calcular erros
            errors = np.abs(y_test_array - y_pred_test_array)
            percent_errors = (errors / (y_test_array + 1e-10)) * 100  # Adicionar epsilon para evitar divisão por zero

            # Criar quartis para faixas
            q1, q2, q3 = np.percentile(y_test_array, [25, 50, 75])

            def get_range_label(value):
                if value <= q1:
                    return f'Baixo (≤{q1:.1f})'
                elif value <= q2:
                    return f'Médio-Baixo ({q1:.1f}-{q2:.1f})'
                elif value <= q3:
                    return f'Médio-Alto ({q2:.1f}-{q3:.1f})'
                else:
                    return f'Alto (≥{q3:.1f})'

            ranges = [get_range_label(v) for v in y_test_array]

            # Criar DataFrame para análise
            error_df = pd.DataFrame({
                'Valor Real': y_test_array,
                'Erro Absoluto': errors,
                'Erro Percentual (%)': percent_errors,
                'Faixa': ranges
            })

            # Gráfico de barras com erro médio por faixa
            error_by_range = error_df.groupby('Faixa')['Erro Absoluto'].agg(['mean', 'std']).reset_index()
            error_by_range = error_by_range.sort_values('mean')

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=error_by_range['mean'],
                y=error_by_range['Faixa'],
                orientation='h',
                name='Erro Médio',
                marker=dict(color='blue'),
                error_x=dict(type='data', array=error_by_range['std'], visible=True),
                hovertemplate='<b>%{y}</b><br>Erro Médio: %{x:.2f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"Erro Médio por Faixa de Valores - {target_name}",
                xaxis_title="Erro Absoluto Médio",
                yaxis_title="Faixa de Valores",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"📌 Barras mostram o erro absoluto médio em cada faixa de {target_name}. Barras de erro indicam o desvio padrão.")

            # Boxplot de erros por faixa
            fig2 = px.box(
                error_df,
                x='Faixa',
                y='Erro Percentual (%)',
                title=f"Distribuição do Erro Percentual por Faixa - {target_name}",
                labels={
                    'Faixa': 'Faixa de Valores',
                    'Erro Percentual (%)': 'Erro Percentual (%)'
                },
                color='Faixa'
            )

            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("📌 Boxplot mostra a distribuição dos erros percentuais em cada faixa. Idealmente, os erros devem ser pequenos e consistentes.")

            # Métricas por faixa
            st.markdown("### 📊 Métricas por Faixa")

            cols = st.columns(len(error_by_range))
            for idx, (_, row) in enumerate(error_by_range.iterrows()):
                with cols[idx]:
                    st.metric(
                        row['Faixa'],
                        f"{row['mean']:.2f}",
                        delta=f"±{row['std']:.2f}",
                        delta_color="off"
                    )

        with tab5:
            # Gráfico de Tendência com Intervalo de Confiança
            if len(selected_features) > 0:
                first_feature = selected_features[0]
                first_feature_name = feature_options.get(first_feature, first_feature)
                target_name = target_options[target_var]

                # Criar dados para a linha de tendência
                feature_values = X[first_feature].values
                feature_min, feature_max = feature_values.min(), feature_values.max()
                feature_range = np.linspace(feature_min, feature_max, 100)

                # Criar matriz de features para predição (usando médias para outras features)
                X_trend = np.zeros((len(feature_range), len(selected_features)))
                for i, feat in enumerate(selected_features):
                    if feat == first_feature:
                        X_trend[:, i] = feature_range
                    else:
                        X_trend[:, i] = X[feat].mean()

                # Predizer valores
                y_trend = model.predict(X_trend)

                # Calcular intervalo de confiança (95%)
                from scipy import stats

                # Calcular o erro padrão da predição
                residuals = y_train - y_pred_train
                mse = np.mean(residuals**2)
                n = len(y_train)
                p = len(selected_features)

                # Graus de liberdade
                df_error = n - p - 1

                # t-statistic para 95% de confiança
                t_stat = stats.t.ppf(0.975, df_error)

                # Erro padrão (aproximação simplificada)
                se = np.sqrt(mse * (1 + 1/n))
                ci_margin = t_stat * se

                ci_lower = y_trend - ci_margin
                ci_upper = y_trend + ci_margin

                # Criar gráfico
                fig = go.Figure()

                # Adicionar intervalo de confiança
                fig.add_trace(go.Scatter(
                    x=feature_range,
                    y=ci_upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))

                fig.add_trace(go.Scatter(
                    x=feature_range,
                    y=ci_lower,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(68, 68, 68, 0.2)',
                    line=dict(width=0),
                    name='Intervalo de Confiança (95%)',
                    hovertemplate=f'<b>{first_feature_name}:</b> %{{x:.2f}}<br><b>IC Inferior:</b> %{{y:.2f}}<extra></extra>'
                ))

                # Adicionar linha de tendência
                fig.add_trace(go.Scatter(
                    x=feature_range,
                    y=y_trend,
                    mode='lines',
                    name='Predição Média',
                    line=dict(color='blue', width=3),
                    hovertemplate=f'<b>{first_feature_name}:</b> %{{x:.2f}}<br><b>Predição:</b> %{{y:.2f}}<extra></extra>'
                ))

                # Adicionar pontos reais
                fig.add_trace(go.Scatter(
                    x=X[first_feature].values,
                    y=y.values,
                    mode='markers',
                    name='Observações Reais',
                    marker=dict(size=6, color='red', opacity=0.5),
                    hovertemplate=f'<b>{first_feature_name}:</b> %{{x:.2f}}<br><b>{target_name}:</b> %{{y:.2f}}<extra></extra>'
                ))

                fig.update_layout(
                    title=f"Tendência com Intervalo de Confiança (95%): {target_name} vs {first_feature_name}",
                    xaxis_title=first_feature_name,
                    yaxis_title=target_name,
                    height=500,
                    showlegend=True,
                    hovermode='closest'
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"📌 A linha azul mostra a predição média em função de {first_feature_name}. A área sombreada representa o intervalo de confiança de 95%, indicando onde estão 95% das predições esperadas.")

def logistic_regression_analysis(players_df, games_df):
    """Análise de Regressão Logística"""
    st.subheader("🎲 Regressão Logística")
    st.markdown("**Predição de categorias (Será que o jogador fará mais de X pontos?)**")
    
    df = games_df.copy()
    df = df.dropna()
    
    if len(df) < 10:
        st.warning("Dados insuficientes para análise de regressão logística.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Configuração da Análise")
        
        classification_options = {
            'vitoria': 'O time vencerá o jogo?',
            'pontos_altos': 'O time fará mais de 110 pontos?',
            'assistencias_altas': 'O time fará mais de 25 assistências?',
            'arremesso_eficiente': 'O time terá mais de 45% nos arremessos?'
        }
        
        target_type = st.selectbox(
            "Tipo de Predição:",
            options=list(classification_options.keys()),
            format_func=lambda x: classification_options[x]
        )
        
        if target_type == 'vitoria':
            df['target'] = df['resultado']
        elif target_type == 'pontos_altos':
            threshold = st.slider("Limite de pontos:", 100, 130, 110)
            df['target'] = (df['pontos'] > threshold).astype(int)
        elif target_type == 'assistencias_altas':
            threshold = st.slider("Limite de assistências:", 15, 35, 25)
            df['target'] = (df['assistencias'] > threshold).astype(int)
        elif target_type == 'arremesso_eficiente':
            threshold = st.slider("Limite de eficiência (%):", 35, 55, 45)
            df['target'] = (df['porcentagem-arremessos'] > threshold/100).astype(int)
        
        feature_options = {
            'arremessos-tentados': 'Arremessos Tentados',
            'arremessos-convertidos': 'Arremessos Convertidos',
            'triplos-tentados': 'Triplos Tentados',
            'triplos-convertidos': 'Triplos Convertidos',
            'lances-livres-tentados': 'Lances Livres Tentados',
            'rebotes-totais': 'Rebotes Totais',
            'assistencias': 'Assistências',
            'roubos': 'Roubos',
            'tocos': 'Tocos',
            'mando-de-jogo': 'Mando de Jogo'
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [col for col in feature_options.keys() if col in numeric_cols]
        
        selected_features = st.multiselect(
            "Variáveis Independentes (X):",
            options=available_features,
            default=available_features[:4] if len(available_features) >= 4 else available_features,
            format_func=lambda x: feature_options.get(x, x)
        )
        
        test_size = st.slider(
            "Porcentagem para teste (%):",
            min_value=10,
            max_value=40,
            value=20
        )
    
    with col2:
        st.subheader("📊 Distribuição da Variável Target")
        
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            
            fig = px.pie(
                values=target_counts.values,
                names=['Não', 'Sim'],
                title=f"Distribuição: {classification_options[target_type]}",
                color_discrete_map={0: 'lightcoral', 1: 'lightblue'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Total de registros:** {len(df)}")
            st.write(f"**Classe positiva:** {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
            st.write(f"**Classe negativa:** {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    if 'target' in df.columns and selected_features and len(selected_features) > 0:
        st.markdown("---")
        
        X = df[selected_features]
        y = df['target']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size/100, random_state=42
        )
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Métricas do Modelo")
            st.metric("Acurácia (Treino)", f"{train_accuracy:.3f}")
            st.metric("Acurácia (Teste)", f"{test_accuracy:.3f}")
            
            cm = confusion_matrix(y_test, y_pred_test)
            st.write("**Matriz de Confusão:**")
            st.write(pd.DataFrame(cm, 
                                index=['Real: Não', 'Real: Sim'],
                                columns=['Pred: Não', 'Pred: Sim']))
        
        with col2:
            st.subheader("⚙️ Coeficientes")
            coef_df = pd.DataFrame({
                'Variável': [feature_options.get(f, f) for f in selected_features],
                'Coeficiente': model.coef_[0],
                'Odds Ratio': np.exp(model.coef_[0])
            })
            st.dataframe(coef_df.round(4), use_container_width=True)
            
            st.write(f"**Intercepto:** {model.intercept_[0]:.4f}")
        
        with col3:
            st.subheader("🔮 Fazer Predição")
            st.write("Insira valores para fazer uma predição:")
            
            prediction_values = {}
            original_features = df[selected_features]
            
            for i, feature in enumerate(selected_features):
                mean_val = original_features[feature].mean()
                min_val = float(original_features[feature].min())
                max_val = float(original_features[feature].max())
                
                prediction_values[feature] = st.number_input(
                    feature_options.get(feature, feature),
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"log_pred_{feature}"
                )
            
            if st.button("🎯 Fazer Predição", key="logistic_predict"):
                pred_input = np.array([list(prediction_values.values())])
                pred_input_scaled = scaler.transform(pred_input)
                
                prediction = model.predict(pred_input_scaled)[0]
                probability = model.predict_proba(pred_input_scaled)[0, 1]
                
                result = "SIM" if prediction == 1 else "NÃO"
                st.success(f"**Predição:** {result}")
                st.info(f"**Probabilidade:** {probability:.1%}")
        
        st.markdown("---")
        st.subheader("📊 Visualizações")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dispersão com Curva Logística",
            "🎯 Predição vs Realidade",
            "🔄 Matriz de Confusão",
            "📈 Curva ROC",
            "📉 Tendência com Intervalo de Confiança"
        ])
        
        with tab1:
            # Diagrama de Dispersão com Curva Logística
            if len(selected_features) > 0:
                first_feature = selected_features[0]
                first_feature_name = feature_options.get(first_feature, first_feature)

                # Dados originais (não normalizados)
                scatter_df = pd.DataFrame({
                    'Variável': df[first_feature].iloc[X_train.shape[0]:],
                    'Classe Real': y_test.map({0: 'Não', 1: 'Sim'}),
                    'Probabilidade': y_pred_proba_test
                })

                # Criar gráfico de dispersão
                fig = px.scatter(
                    scatter_df,
                    x='Variável',
                    y='Probabilidade',
                    color='Classe Real',
                    title=f"Dispersão com Curva Logística: {first_feature_name}",
                    labels={
                        'Variável': first_feature_name,
                        'Probabilidade': 'Probabilidade Predita (Classe = Sim)',
                        'Classe Real': 'Classe Real'
                    },
                    color_discrete_map={'Não': 'red', 'Sim': 'blue'}
                )

                # Adicionar linha em 0.5 (threshold de decisão)
                fig.add_hline(
                    y=0.5,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Threshold de Decisão (0.5)",
                    annotation_position="right"
                )

                fig.update_layout(
                    height=500,
                    xaxis_title=first_feature_name,
                    yaxis_title="Probabilidade de Classe = Sim",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"📌 Pontos acima da linha verde são classificados como 'Sim', abaixo como 'Não'.")

        with tab2:
            # Gráfico de Predição vs. Realidade
            pred_real_df = pd.DataFrame({
                'Índice': range(len(y_test)),
                'Real': y_test.map({0: 'Não', 1: 'Sim'}),
                'Predito': [('Sim' if p == 1 else 'Não') for p in y_pred_test],
                'Correto': y_test == y_pred_test
            })

            fig = go.Figure()

            # Adicionar valores reais
            fig.add_trace(go.Scatter(
                x=pred_real_df['Índice'],
                y=y_test,
                mode='markers',
                name='Valor Real',
                marker=dict(size=10, color='blue', symbol='circle'),
                text=pred_real_df['Real'],
                hovertemplate='<b>Real:</b> %{text}<br><b>Índice:</b> %{x}<extra></extra>'
            ))

            # Adicionar valores preditos
            fig.add_trace(go.Scatter(
                x=pred_real_df['Índice'],
                y=y_pred_test,
                mode='markers',
                name='Predição',
                marker=dict(
                    size=8,
                    color=['green' if c else 'red' for c in pred_real_df['Correto']],
                    symbol='x'
                ),
                text=pred_real_df['Predito'],
                hovertemplate='<b>Predito:</b> %{text}<br><b>Índice:</b> %{x}<extra></extra>'
            ))

            fig.update_layout(
                title="Comparação: Valores Reais vs Predições",
                xaxis_title="Índice da Amostra (Conjunto de Teste)",
                yaxis_title="Classe (0 = Não, 1 = Sim)",
                yaxis=dict(tickmode='linear', tick0=0, dtick=1),
                height=500,
                showlegend=True,
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"📌 Marcas verdes indicam predições corretas, vermelhas indicam erros.")

            # Adicionar tabela de resumo
            correct_count = pred_real_df['Correto'].sum()
            total_count = len(pred_real_df)
            st.info(f"**Acurácia no conjunto de teste:** {correct_count}/{total_count} ({test_accuracy:.1%})")

        with tab3:
            # Matriz de Confusão
            cm = confusion_matrix(y_test, y_pred_test)

            fig = px.imshow(
                cm,
                title="Matriz de Confusão - Desempenho do Modelo",
                labels=dict(x="Classe Predita", y="Classe Real", color="Quantidade"),
                x=['Não', 'Sim'],
                y=['Não', 'Sim'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig.update_layout(
                height=500,
                xaxis_title="Classe Predita",
                yaxis_title="Classe Real"
            )
            fig.update_traces(
                hovertemplate="<b>Real:</b> %{y}<br><b>Predito:</b> %{x}<br><b>Quantidade:</b> %{z}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 Diagonal principal (azul escuro) representa predições corretas.")

            # Adicionar métricas da matriz de confusão
            tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Verdadeiros Positivos", tp)
            with col2:
                st.metric("Verdadeiros Negativos", tn)
            with col3:
                st.metric("Falsos Positivos", fp)
            with col4:
                st.metric("Falsos Negativos", fn)

        with tab4:
            # Curva ROC (Receiver Operating Characteristic)
            from sklearn.metrics import roc_curve, roc_auc_score

            # Calcular a curva ROC
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
            auc_score = roc_auc_score(y_test, y_pred_proba_test)

            # Criar o gráfico
            fig = go.Figure()

            # Adicionar a curva ROC
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'Curva ROC (AUC = {auc_score:.3f})',
                line=dict(color='blue', width=3),
                hovertemplate='<b>FPR:</b> %{x:.3f}<br><b>TPR:</b> %{y:.3f}<extra></extra>'
            ))

            # Adicionar linha diagonal (classificador aleatório)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Classificador Aleatório (AUC = 0.5)',
                line=dict(dash='dash', color='red', width=2),
                hovertemplate='<b>Linha de Referência</b><extra></extra>'
            ))

            # Adicionar ponto do threshold atual (0.5)
            threshold_idx = np.argmin(np.abs(thresholds - 0.5))
            fig.add_trace(go.Scatter(
                x=[fpr[threshold_idx]],
                y=[tpr[threshold_idx]],
                mode='markers',
                name='Threshold = 0.5',
                marker=dict(size=12, color='green', symbol='star'),
                hovertemplate=f'<b>Threshold = 0.5</b><br>FPR: {fpr[threshold_idx]:.3f}<br>TPR: {tpr[threshold_idx]:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"Curva ROC (Receiver Operating Characteristic)<br>Área Sob a Curva (AUC) = {auc_score:.3f}",
                xaxis_title="Taxa de Falsos Positivos (FPR) - False Positive Rate",
                yaxis_title="Taxa de Verdadeiros Positivos (TPR) - True Positive Rate",
                height=500,
                showlegend=True,
                hovermode='closest',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )

            st.plotly_chart(fig, use_container_width=True)

            # Interpretação do AUC
            if auc_score >= 0.9:
                interpretation = "🌟 Excelente! O modelo tem um desempenho muito alto."
                color = "success"
            elif auc_score >= 0.8:
                interpretation = "✅ Bom! O modelo tem um desempenho satisfatório."
                color = "success"
            elif auc_score >= 0.7:
                interpretation = "⚠️ Razoável. O modelo tem um desempenho aceitável, mas pode ser melhorado."
                color = "warning"
            else:
                interpretation = "❌ Fraco. O modelo precisa de melhorias significativas."
                color = "error"

            if color == "success":
                st.success(interpretation)
            elif color == "warning":
                st.warning(interpretation)
            else:
                st.error(interpretation)

            st.caption("📌 A Curva ROC mostra a relação entre TPR (sensibilidade) e FPR (1-especificidade). Quanto mais próxima do canto superior esquerdo, melhor o modelo. AUC = 1.0 representa um classificador perfeito, AUC = 0.5 representa um classificador aleatório.")

            # Métricas adicionais
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AUC Score", f"{auc_score:.3f}")
            with col2:
                # Sensibilidade (TPR no threshold 0.5)
                sensitivity = tpr[threshold_idx]
                st.metric("Sensibilidade (TPR)", f"{sensitivity:.3f}")
            with col3:
                # Especificidade
                specificity = 1 - fpr[threshold_idx]
                st.metric("Especificidade", f"{specificity:.3f}")

        with tab5:
            # Gráfico de Tendência com Intervalo de Confiança
            if len(selected_features) > 0:
                first_feature = selected_features[0]
                first_feature_name = feature_options.get(first_feature, first_feature)

                # Criar dados para a curva de tendência
                feature_values = df[first_feature].values
                feature_min, feature_max = feature_values.min(), feature_values.max()
                feature_range = np.linspace(feature_min, feature_max, 100)

                # Criar matriz de features para predição (usando médias para outras features)
                X_trend = np.zeros((len(feature_range), len(selected_features)))
                for i, feat in enumerate(selected_features):
                    if feat == first_feature:
                        X_trend[:, i] = feature_range
                    else:
                        X_trend[:, i] = df[feat].mean()

                # Normalizar
                X_trend_scaled = scaler.transform(X_trend)

                # Predizer probabilidades
                proba_trend = model.predict_proba(X_trend_scaled)[:, 1]

                # Calcular intervalo de confiança (aproximado usando bootstrap simples)
                from scipy import stats
                confidence_level = 0.95
                z_score = stats.norm.ppf((1 + confidence_level) / 2)

                # Desvio padrão estimado
                std_error = np.sqrt(proba_trend * (1 - proba_trend) / len(y_train))
                ci_lower = proba_trend - z_score * std_error
                ci_upper = proba_trend + z_score * std_error

                # Clipar para [0, 1]
                ci_lower = np.clip(ci_lower, 0, 1)
                ci_upper = np.clip(ci_upper, 0, 1)

                # Criar gráfico
                fig = go.Figure()

                # Adicionar intervalo de confiança
                fig.add_trace(go.Scatter(
                    x=feature_range,
                    y=ci_upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))

                fig.add_trace(go.Scatter(
                    x=feature_range,
                    y=ci_lower,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(68, 68, 68, 0.2)',
                    line=dict(width=0),
                    name=f'Intervalo de Confiança ({confidence_level:.0%})',
                    hovertemplate=f'<b>{first_feature_name}:</b> %{{x:.2f}}<br><b>IC Inferior:</b> %{{y:.2%}}<extra></extra>'
                ))

                # Adicionar linha de tendência
                fig.add_trace(go.Scatter(
                    x=feature_range,
                    y=proba_trend,
                    mode='lines',
                    name='Probabilidade Predita',
                    line=dict(color='blue', width=3),
                    hovertemplate=f'<b>{first_feature_name}:</b> %{{x:.2f}}<br><b>Probabilidade:</b> %{{y:.2%}}<extra></extra>'
                ))

                # Adicionar pontos reais
                real_proba_df = pd.DataFrame({
                    'feature': df[first_feature].iloc[len(y_train):].values,
                    'real': y_test.values
                })

                fig.add_trace(go.Scatter(
                    x=real_proba_df['feature'],
                    y=real_proba_df['real'],
                    mode='markers',
                    name='Observações Reais',
                    marker=dict(size=6, color='red', opacity=0.5),
                    hovertemplate=f'<b>{first_feature_name}:</b> %{{x:.2f}}<br><b>Classe Real:</b> %{{y}}<extra></extra>'
                ))

                fig.update_layout(
                    title=f"Tendência de Probabilidade com Intervalo de Confiança ({confidence_level:.0%})",
                    xaxis_title=first_feature_name,
                    yaxis_title="Probabilidade de Classe = Sim",
                    height=500,
                    showlegend=True,
                    hovermode='closest'
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"📌 A linha azul mostra a tendência da probabilidade em função de {first_feature_name}. A área sombreada representa o intervalo de confiança de {confidence_level:.0%}.")

def prediction_interface(players_df, games_df):
    """Interface para predições específicas"""
    st.header("🎯 Predições Específicas")
    st.write("")

    st.markdown("**Faça perguntas específicas sobre desempenho de jogadores e do time**")
    
    
    if not SKLEARN_AVAILABLE:
        st.error("⚠️ Scikit-learn não está instalado. Instale com: pip install scikit-learn")
        return
    
    prediction_type = st.selectbox(
        "Tipo de Predição",
        ["Desempenho do Jogador", "Desempenho do Time"],
        help="Escolha se quer prever algo sobre um jogador específico ou sobre o time"
    )
    
    if prediction_type == "Desempenho do Jogador":
        player_specific_predictions(players_df)
    else:
        team_specific_predictions(games_df)

def player_specific_predictions(players_df):
    """Predições específicas para jogadores"""
    st.subheader("👤 Predições de Jogadores")

    active_players = players_df[players_df['jogos-disputados_total'] >= 5].copy()
    
    if len(active_players) == 0:
        st.warning("Não há jogadores com dados suficientes para predição.")
        return
    
    if 'selected_player_idx' not in st.session_state:
        st.session_state.selected_player_idx = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        player_options = {}
        for idx, row in active_players.iterrows():
            pos_name = {1: 'Guard', 2: 'Forward', 3: 'Forward-Center', 4: 'Center-Forward', 5: 'Center'}
            position = pos_name.get(row['posicao-g-f-fc-cf-c'], f"Pos-{row['posicao-g-f-fc-cf-c']}")
            
            if 'nome-jogador' in row and pd.notna(row['nome-jogador']) and row['nome-jogador'].strip():
                player_name = f"{row['nome-jogador']} ({position})"
            else:
                player_name = f"{position} #{idx} ({row['idade']} anos)"
            
            player_options[player_name] = idx
        
        selected_player_name = st.selectbox(
            "Selecione o Jogador:",
            options=list(player_options.keys()),
            help="Escolha o jogador para fazer a predição",
            key="player_selector"
        )
        
        if selected_player_name in player_options:
            selected_player_idx = player_options[selected_player_name]
            
            if selected_player_idx in active_players.index:
                player_data = active_players.loc[selected_player_idx]
                st.session_state.selected_player_idx = selected_player_idx
            else:
                st.error("Erro: Jogador selecionado não encontrado nos dados.")
                return
        else:
            st.error("Erro: Jogador selecionado inválido.")
            return
        
        stat_type = st.selectbox(
            "O que queremos prever?",
            ["Pontos", "Rebotes", "Assistências"],
            help="Escolha a estatística que quer prever",
            key="stat_type_selector"
        )
        
        if stat_type == "Pontos":
            current_avg = player_data['pontos_media']
            target_value = st.number_input(
                f"Quantos {stat_type.lower()} o jogador fará?",
                min_value=0,
                max_value=100,
                value=max(0, min(100, int(current_avg))),
                help=f"Média atual: {current_avg:.1f} por jogo",
                key=f"points_input_{selected_player_idx}"
            )
            stat_column = 'pontos_media'
        elif stat_type == "Rebotes":
            current_avg = player_data['rebotes-totais_media']
            target_value = st.number_input(
                f"Quantos {stat_type.lower()} o jogador fará?",
                min_value=0,
                max_value=30,
                value=max(0, min(30, int(current_avg))),
                help=f"Média atual: {current_avg:.1f} por jogo",
                key=f"rebounds_input_{selected_player_idx}"
            )
            stat_column = 'rebotes-totais_media'
        else:
            current_avg = player_data['assistencias_media']
            target_value = st.number_input(
                f"Quantas {stat_type.lower()} o jogador fará?",
                min_value=0,
                max_value=20,
                value=max(0, min(20, int(current_avg))),
                help=f"Média atual: {current_avg:.1f} por jogo",
                key=f"assists_input_{selected_player_idx}"
            )
            stat_column = 'assistencias_media'
    
    with col2:
        st.subheader("📊 Dados do Jogador Selecionado")
        
        st.write("**Informações do Jogador:**")
        player_info_df = pd.DataFrame({
            'Variável': [
                'Nome',
                'Posição', 
                'Idade',
                'Jogos Disputados',
                'Minutos por Jogo',
                'Pontos por Jogo',
                'Rebotes por Jogo',
                'Assistências por Jogo',
                '% Arremessos'
            ],
            'Valor': [
                player_data.get('nome-jogador', 'N/A') if pd.notna(player_data.get('nome-jogador')) else 'N/A',
                {1: 'Guard', 2: 'Forward', 3: 'Forward-Center', 4: 'Center-Forward', 5: 'Center'}.get(player_data['posicao-g-f-fc-cf-c'], 'N/A'),
                f"{player_data['idade']} anos",
                f"{player_data['jogos-disputados_total']} jogos",
                f"{player_data['minutos_media']:.1f} min",
                f"{player_data['pontos_media']:.1f} pts",
                f"{player_data['rebotes-totais_media']:.1f} reb",
                f"{player_data['assistencias_media']:.1f} ast",
                f"{player_data['porcentagem-arremessos_media']*100:.1f}%"
            ]
        })
        st.dataframe(player_info_df, use_container_width=True)
    
    if st.button("🔮 Fazer Predição", type="primary", key=f"predict_button_{selected_player_idx}_{stat_type}"):
        make_player_prediction(active_players, selected_player_idx, stat_column, target_value, stat_type)

def make_player_prediction(players_df, player_idx, stat_column, target_value, stat_type):
    """Faz a predição para um jogador específico"""

    feature_columns = [
        'idade', 'jogos-disputados_total', 'minutos_media', 'arremessos-tentados_media',
        'porcentagem-arremessos_media', 'rebotes-totais_media', 'assistencias_media'
    ]

    available_features = [col for col in feature_columns if col in players_df.columns and col != stat_column]

    if len(available_features) < 3:
        st.error("Dados insuficientes para fazer a predição.")
        return

    X = players_df[available_features].fillna(0)
    y = players_df[stat_column].fillna(0)

    model = LinearRegression()
    model.fit(X, y)

    player_features = players_df.loc[player_idx, available_features].values.reshape(1, -1)

    predicted_per_game = model.predict(player_features)[0]
    player_data = players_df.loc[player_idx]
    
    probability = 50

    diff_from_prediction = abs(target_value - predicted_per_game)
    if diff_from_prediction <= 1:
        probability += 30
    elif diff_from_prediction <= 2:
        probability += 20
    elif diff_from_prediction <= 3:
        probability += 10
    elif diff_from_prediction <= 5:
        probability -= 10
    else:
        probability -= 20
    
    similar_position = players_df[
        players_df['posicao-g-f-fc-cf-c'] == player_data['posicao-g-f-fc-cf-c']
    ]
    
    if len(similar_position) > 1:
        position_avg = similar_position[stat_column].mean()
        position_std = similar_position[stat_column].std()
        
        if position_std > 0:
            z_score = abs(target_value - position_avg) / position_std
            if z_score <= 1:
                probability += 15  
            elif z_score <= 2:
                probability += 5  
            else:
                probability -= 15  
    
    current_avg = player_data[stat_column]
    trend_factor = abs(target_value - current_avg) / (current_avg + 0.1) 
    
    if trend_factor <= 0.1: 
        probability += 20
    elif trend_factor <= 0.2: 
        probability += 10
    elif trend_factor <= 0.5: 
        probability -= 5
    else:  
        probability -= 15
    
    if player_data['jogos-disputados_total'] >= 20:
        probability += 5  
    elif player_data['jogos-disputados_total'] <= 5:
        probability -= 10  
    
    if stat_type == "Pontos":
        if 'porcentagem-arremessos_media' in player_data:
            shoot_pct = player_data['porcentagem-arremessos_media']
            if shoot_pct > 0.5:  
                probability += 5
            elif shoot_pct < 0.4:
                probability -= 5
    elif stat_type == "Rebotes":
        if player_data['posicao-g-f-fc-cf-c'] >= 4:  
            probability += 5
    elif stat_type == "Assistências":
        if player_data['posicao-g-f-fc-cf-c'] <= 2:  
            probability += 5
    
    probability = max(5, min(95, probability)) 

    st.markdown("---")
    st.subheader("🎯 Resultado da Predição")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"Predição Atual do Modelo",
            f"{predicted_per_game:.1f} {stat_type.lower()}/jogo",
            help="Baseado no desempenho histórico do jogador"
        )
    
    with col2:
        comparison = "⬆️" if target_value > predicted_per_game else "⬇️" if target_value < predicted_per_game else "➡️"
        st.metric(
            f"Meta Desejada",
            f"{target_value} {stat_type.lower()}/jogo",
            delta=f"{comparison} {abs(target_value - predicted_per_game):.1f}"
        )
    
    with col3:
        probability_color = "🟢" if probability > 70 else "🟡" if probability > 40 else "🔴"
        st.metric(
            "Probabilidade de Sucesso",
            f"{probability_color} {probability:.0f}%",
            help="Baseado em jogadores similares"
        )
    
    if probability > 70:
        interpretation = "🎉 **Alta probabilidade!** O jogador tem boas chances de atingir essa meta."
    elif probability > 40:
        interpretation = "⚠️ **Probabilidade moderada.** A meta é desafiadora mas possível."
    else:
        interpretation = "🚨 **Baixa probabilidade.** A meta é muito ambiciosa para o perfil atual do jogador."
    
    st.markdown(f"**Interpretação:** {interpretation}")
    
    with st.expander("🔍 Ver detalhes do cálculo da probabilidade"):
        st.markdown("**Fatores considerados no cálculo:**")
        st.markdown(f"• **Predição do modelo:** {predicted_per_game:.1f} {stat_type.lower()}/jogo")
        st.markdown(f"• **Meta desejada:** {target_value} {stat_type.lower()}/jogo")
        st.markdown(f"• **Diferença:** {abs(target_value - predicted_per_game):.1f}")
        st.markdown(f"• **Média atual do jogador:** {player_data[stat_column]:.1f} {stat_type.lower()}/jogo")
        st.markdown(f"• **Jogos disputados:** {player_data['jogos-disputados_total']}")
        st.markdown(f"• **Posição:** {player_data['posicao-g-f-fc-cf-c']}")
        
        similar_position = players_df[
            players_df['posicao-g-f-fc-cf-c'] == player_data['posicao-g-f-fc-cf-c']
        ]
        if len(similar_position) > 1:
            position_avg = similar_position[stat_column].mean()
            st.markdown(f"• **Média da posição:** {position_avg:.1f} {stat_type.lower()}/jogo")
            st.markdown(f"• **Diferença da posição:** {abs(target_value - position_avg):.1f}")
        
        st.markdown("**Probabilidade calculada dinamicamente baseada em:**")
        st.markdown("- Proximidade da predição do modelo")
        st.markdown("- Comparação com jogadores da mesma posição")
        st.markdown("- Tendência histórica do jogador")
        st.markdown("- Experiência (jogos disputados)")
        st.markdown("- Características específicas da estatística")

def team_specific_predictions(games_df):
    """Predições específicas para o time"""
    st.subheader("🏀 Predições do Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team_stat = st.selectbox(
            "O que queremos prever para o time?",
            ["Pontos", "Rebotes", "Assistências"],
            help="Escolha a estatística do time que quer prever",
            key="team_stat_selector"
        )
        
        if team_stat == "Pontos":
            current_avg = games_df['pontos'].mean()
            target_value = st.number_input(
                f"Quantos {team_stat.lower()} o time fará no próximo jogo?",
                min_value=60,
                max_value=150,
                value=int(current_avg),
                help=f"Média atual: {current_avg:.1f} por jogo",
                key="team_points_input"
            )
            stat_column = 'pontos'
        elif team_stat == "Rebotes":
            current_avg = games_df['rebotes-totais'].mean()
            target_value = st.number_input(
                f"Quantos {team_stat.lower()} o time fará no próximo jogo?",
                min_value=20,
                max_value=80,
                value=int(current_avg),
                help=f"Média atual: {current_avg:.1f} por jogo",
                key="team_rebounds_input"
            )
            stat_column = 'rebotes-totais'
        else: 
            current_avg = games_df['assistencias'].mean()
            target_value = st.number_input(
                f"Quantas {team_stat.lower()} o time fará no próximo jogo?",
                min_value=10,
                max_value=40,
                value=int(current_avg),
                help=f"Média atual: {current_avg:.1f} por jogo",
                key="team_assists_input"
            )
            stat_column = 'assistencias'
        
        game_context = st.selectbox(
            "Contexto do jogo:",
            ["Casa", "Fora"],
            help="O time joga em casa ou fora?",
            key="game_context_selector"
        )
    
    with col2:
        st.subheader("📊 Estatísticas Atuais do Time")
        
        team_stats = {
            'Jogos Disputados': len(games_df),
            'Vitórias': f"{games_df['resultado'].sum()} ({games_df['resultado'].mean()*100:.1f}%)",
            'Pontos/Jogo': f"{games_df['pontos'].mean():.1f}",
            'Rebotes/Jogo': f"{games_df['rebotes-totais'].mean():.1f}",
            'Assistências/Jogo': f"{games_df['assistencias'].mean():.1f}",
            '% Arremessos': f"{games_df['porcentagem-arremessos'].mean()*100:.1f}%",
            'Em Casa': f"{games_df[games_df['mando-de-jogo']==1]['resultado'].mean()*100:.1f}% vitórias",
            'Fora': f"{games_df[games_df['mando-de-jogo']==0]['resultado'].mean()*100:.1f}% vitórias"
        }
        
        for stat_name, stat_value in team_stats.items():
            st.metric(stat_name, stat_value)
    
    if st.button("🔮 Fazer Predição do Time", type="primary", key=f"team_predict_button_{team_stat}_{game_context}"):
        make_team_prediction(games_df, stat_column, target_value, team_stat, game_context)

def make_team_prediction(games_df, stat_column, target_value, stat_type, game_context):
    """Faz a predição para o time"""
    
    context_value = 1 if game_context == "Casa" else 0
    context_games = games_df[games_df['mando-de-jogo'] == context_value]
    
    if len(context_games) < 3:
        context_games = games_df  
    
    context_avg = context_games[stat_column].mean()
    context_std = context_games[stat_column].std()
    overall_avg = games_df[stat_column].mean()
    overall_std = games_df[stat_column].std()
    
    probability = 50
    
    if context_std > 0:
        z_score = abs(target_value - context_avg) / context_std
        if z_score <= 0.5:
            probability += 25 
        elif z_score <= 1:
            probability += 15  
        elif z_score <= 1.5:
            probability += 5  
        elif z_score <= 2:
            probability -= 10 
        else:
            probability -= 25  
    
    recent_games = games_df.tail(5)
    if len(recent_games) >= 3:
        recent_avg = recent_games[stat_column].mean()
        recent_context = recent_games[recent_games['mando-de-jogo'] == context_value]
        
        if len(recent_context) >= 2:
            recent_context_avg = recent_context[stat_column].mean()
            if abs(target_value - recent_context_avg) < abs(target_value - context_avg):
                probability += 10
        
        if abs(target_value - recent_avg) <= overall_std:
            probability += 5
    
    home_games = games_df[games_df['mando-de-jogo'] == 1]
    away_games = games_df[games_df['mando-de-jogo'] == 0]
    
    if len(home_games) >= 3 and len(away_games) >= 3:
        home_avg = home_games[stat_column].mean()
        away_avg = away_games[stat_column].mean()
        
        if game_context == "Casa" and home_avg > away_avg:
            probability += 8 
        elif game_context == "Fora" and away_avg > home_avg:
            probability += 8  
        elif game_context == "Casa" and home_avg < away_avg:
            probability -= 5
        elif game_context == "Fora" and away_avg < home_avg:
            probability -= 5 
    
    if overall_std > 0:
        coefficient_variation = overall_std / overall_avg
        if coefficient_variation < 0.15:
            probability += 10
        elif coefficient_variation > 0.30:
            probability -= 8
    
    if stat_type == "Pontos":
        avg_fg_pct = games_df['porcentagem-arremessos'].mean()
        if avg_fg_pct > 0.47: 
            probability += 5
        elif avg_fg_pct < 0.42:
            probability -= 5
            
    elif stat_type == "Rebotes":
        context_reb_avg = context_games['rebotes-totais'].mean() if 'rebotes-totais' in context_games.columns else 0
        if target_value <= context_reb_avg * 1.1: 
            probability += 8
            
    elif stat_type == "Assistências":
        context_ast_avg = context_games['assistencias'].mean()
        if target_value >= context_ast_avg * 0.9: 
            probability += 5
    
    total_games = len(games_df)
    if total_games >= 20:
        probability += 3  
    elif total_games <= 5:
        probability -= 5  
    
    probability = max(5, min(95, probability))
    
    st.markdown("---")
    st.subheader("🎯 Resultado da Predição do Time")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"Média {game_context}",
            f"{context_avg:.1f} {stat_type.lower()}",
            help=f"Desempenho médio do time jogando {game_context.lower()}"
        )
    
    with col2:
        comparison = "⬆️" if target_value > context_avg else "⬇️" if target_value < context_avg else "➡️"
        st.metric(
            f"Meta Desejada",
            f"{target_value} {stat_type.lower()}",
            delta=f"{comparison} {abs(target_value - context_avg):.1f}"
        )
    
    with col3:
        probability_color = "🟢" if probability > 70 else "🟡" if probability > 40 else "🔴"
        st.metric(
            "Probabilidade de Sucesso",
            f"{probability_color} {probability:.0f}%",
            help="Baseado no histórico do time"
        )
    
    if probability > 70:
        interpretation = "🎉 **Alta probabilidade!** O time tem boas chances de atingir essa marca."
    elif probability > 40:
        interpretation = "⚠️ **Probabilidade moderada.** A meta está dentro da variação normal do time."
    else:
        interpretation = "🚨 **Baixa probabilidade.** A meta está fora do padrão histórico do time."
    
    st.markdown(f"**Interpretação:** {interpretation}")
    
    with st.expander("🔍 Ver detalhes do cálculo da probabilidade"):
        st.markdown("**Estatísticas utilizadas no cálculo:**")
        st.markdown(f"• **Média {game_context.lower()}:** {context_avg:.1f} {stat_type.lower()}/jogo")
        if context_std > 0:
            z_score = abs(target_value - context_avg) / context_std
            st.markdown(f"• **Desvio padrão {game_context.lower()}:** {context_std:.1f}")
            st.markdown(f"• **Z-score:** {z_score:.2f} (quantos desvios da média)")
        
        overall_avg = games_df[stat_column].mean()
        st.markdown(f"• **Média geral:** {overall_avg:.1f} {stat_type.lower()}/jogo")
        
        recent_games = games_df.tail(5)
        if len(recent_games) >= 3:
            recent_avg = recent_games[stat_column].mean()
            st.markdown(f"• **Média últimos 5 jogos:** {recent_avg:.1f} {stat_type.lower()}/jogo")
        
        st.markdown(f"• **Total de jogos analisados:** {len(games_df)}")
        st.markdown(f"• **Jogos no contexto {game_context.lower()}:** {len(context_games)}")
        
        home_avg = games_df[games_df['mando-de-jogo'] == 1][stat_column].mean()
        away_avg = games_df[games_df['mando-de-jogo'] == 0][stat_column].mean()
        st.markdown(f"• **Média em casa:** {home_avg:.1f} {stat_type.lower()}/jogo")
        st.markdown(f"• **Média fora:** {away_avg:.1f} {stat_type.lower()}/jogo")
        
        st.markdown("**Fatores considerados no cálculo:**")
        st.markdown("- Proximidade da meta em relação à média histórica")
        st.markdown("- Tendência recente dos últimos jogos")
        st.markdown("- Performance histórica em casa vs fora")
        st.markdown("- Consistência da equipe (variabilidade)")
        st.markdown("- Características específicas da estatística")
        st.markdown("- Quantidade de dados históricos disponíveis")
    
    st.subheader("💡 Fatores que Podem Influenciar")
    
    factors_col1, factors_col2 = st.columns(2)
    
    with factors_col1:
        st.markdown("**Fatores Positivos:**")
        if game_context == "Casa":
            st.markdown("- 🏠 Vantagem de jogar em casa")
        st.markdown("- 📈 Tendência recente do time")
        st.markdown("- 🎯 Motivação da equipe")
        
    with factors_col2:
        st.markdown("**Fatores de Risco:**")
        if game_context == "Fora":
            st.markdown("- ✈️ Desgaste de viagem")
        st.markdown("- 🏥 Possíveis lesões")
        st.markdown("- 🛡️ Qualidade da defesa adversária")

def notebook_regression_analysis(games_df):
    """Análise de Regressão Linear baseada no notebook linear_regression_att.ipynb"""
    st.header("📈 Análise de Regressão Linear - Equação 1")

    if not SKLEARN_AVAILABLE:
        st.error("⚠️ Scikit-learn não está instalado. Instale com: pip install scikit-learn")
        return

    st.markdown("---")

    st.subheader("📐 Equação 1: Modelo de Regressão Linear Múltipla")
    st.latex(r"y = a + b \cdot x")
    st.latex(r"y = \beta_0 + \beta_1x + \varepsilon")
    st.latex(r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \varepsilon")

    st.info("""
    **Interpretação da Equação:**
    - **y**: Variável dependente (o que queremos prever)
    - **β₀ (beta zero)**: Intercepto (valor base quando todas as variáveis independentes são zero)
    - **β₁, β₂, ..., βₙ (betas)**: Coeficientes de regressão (quantificam o impacto de cada variável independente)
    - **x₁, x₂, ..., xₙ**: Variáveis independentes (features que influenciam a predição)
    - **ε (epsilon)**: Termo de erro (variação não explicada pelo modelo)
    """)

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("⚙️ Configuração do Modelo")

        numeric_cols = games_df.select_dtypes(include=[np.number]).columns.tolist()

        exclude_cols = ['data-jogo']
        available_columns = [col for col in numeric_cols if col not in exclude_cols]

        target_variable = st.selectbox(
            "🎯 Variável Dependente (y) - O que queremos prever:",
            options=available_columns,
            index=available_columns.index('pontos') if 'pontos' in available_columns else 0,
            help="Esta é a variável que o modelo tentará prever",
            key="notebook_target_var"
        )

        available_features = [col for col in available_columns if col != target_variable]

        if target_variable == 'pontos':
            leakage_vars = ['saldo-pontos', 'resultado']
            available_features = [col for col in available_features if col not in leakage_vars]

        use_all = st.checkbox("Usar todas as variáveis disponíveis", value=False, key="notebook_use_all")

        if use_all:
            selected_features = available_features
        else:
            default_features = [
                'arremessos-convertidos',
                'porcentagem-arremessos',
                'triplos-convertidos',
                'assistencias'
            ]
            default_features = [f for f in default_features if f in available_features]

            selected_features = st.multiselect(
                "📊 Variáveis Independentes (x₁, x₂, ..., xₙ) - Fatores que influenciam:",
                options=available_features,
                default=default_features,
                help="Estas são as variáveis que o modelo usará para fazer a previsão",
                key="notebook_features"
            )

        test_size = st.slider(
            "Tamanho do conjunto de teste (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Porcentagem dos dados reservada para validar o modelo",
            key="notebook_test_size"
        )

        random_state = st.number_input(
            "Random State (reprodutibilidade)",
            min_value=0,
            max_value=100,
            value=42,
            help="Garante que os resultados sejam reproduzíveis",
            key="notebook_random_state"
        )

    with col2:
        st.subheader("📊 Informações do Dataset")

        st.metric("Total de Amostras", len(games_df))
        st.metric("Variável Dependente", target_variable)
        st.metric("Número de Features Selecionadas", len(selected_features) if selected_features else 0)

        if selected_features:
            st.write("**Features Selecionadas:**")
            
            with st.expander(f"Ver todas as {len(selected_features)} variáveis selecionadas"):
                col_a, col_b = st.columns(2)
                
                for i, feature in enumerate(selected_features):
                    col_idx = i % 2
                    feature_num = i + 1
                    
                    if col_idx == 0:
                        col_a.write(f"{feature_num}. {feature}")
                    else:
                        col_b.write(f"{feature_num}. {feature}")

    if not selected_features or len(selected_features) == 0:
        st.warning("⚠️ Por favor, selecione pelo menos uma variável independente para treinar o modelo.")
        return

    st.markdown("---")

    # Status do modelo
    if 'notebook_model' in st.session_state:
        try:
            model_data = st.session_state['notebook_model']
            if 'target' in model_data and 'features' in model_data:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.success(f"✅ Modelo treinado encontrado! Variável alvo: {model_data['target']} | Features: {len(model_data['features'])}")
                with col2:
                    if st.button("🗑️ Limpar Modelo", key="clear_model_button"):
                        del st.session_state['notebook_model']
                        st.rerun()
            else:
                st.warning("⚠️ Modelo encontrado, mas dados incompletos. Treine novamente.")
                del st.session_state['notebook_model']
        except Exception as e:
            st.error(f"❌ Erro ao carregar modelo: {e}")
            del st.session_state['notebook_model']
    else:
        st.info("ℹ️ Nenhum modelo treinado encontrado. Treine um modelo abaixo.")
    
    if st.button("🚀 Treinar Modelo de Regressão Linear", type="primary", use_container_width=True, key="notebook_train_button"):
        with st.spinner("Treinando modelo..."):
            X = games_df[selected_features]
            y = games_df[target_variable]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            rmse_train = np.sqrt(mse_train)
            rmse_test = np.sqrt(mse_test)

            st.session_state['notebook_model'] = {
                'model': model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'features': selected_features,
                'target': target_variable,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'mse_train': mse_train,
                'mse_test': mse_test,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test
            }

            st.success("✅ Modelo treinado com sucesso!")
            st.rerun()  # Força a atualização da interface

    if 'notebook_model' in st.session_state:
        model_data = st.session_state['notebook_model']
        
        required_keys = ['model', 'features', 'target', 'r2_train', 'r2_test', 'rmse_train', 'rmse_test']
        missing_keys = [key for key in required_keys if key not in model_data]
        
        if missing_keys:
            st.error(f"❌ Dados do modelo incompletos. Chaves faltando: {missing_keys}")
            st.warning("Por favor, treine o modelo novamente.")
            del st.session_state['notebook_model']
            return
            
        model = model_data['model']

        st.markdown("---")
        st.header("📊 Resultados do Modelo")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📐 Equação e Coeficientes",
            "📈 Métricas de Desempenho",
            "🔮 Fazer Previsões",
            "📊 Visualizações"
        ])

        with tab1:
            st.subheader("📐 Equação de Regressão Treinada")

            intercept = model.intercept_
            coefficients = model.coef_

            equation_html = f"""
            <div class="equation-container">
                <span class="equation-part"><strong>{model_data['target']}</strong></span>
            """
            
            if intercept >= 0:
                equation_html += f'<span class="equation-part">= {intercept:.4f}</span>'
            else:
                equation_html += f'<span class="equation-part">= {intercept:.4f}</span>'
            
            for i, (coef, feature) in enumerate(zip(coefficients, model_data['features'])):
                sign = "+" if coef >= 0 else ""
                term_html = f'<span class="equation-part">{sign} {coef:.4f} × {feature}</span>'
                equation_html += term_html
            
            equation_html += '<span class="equation-part">+ ε</span></div>'
            
            equation_css = """
            <style>
            .equation-container {
                font-family: 'Computer Modern', 'Latin Modern Math', 'Times New Roman', serif;
                font-size: 1.2em;
                line-height: 1.8;
                padding: 1rem;
                margin: 1rem 0;
                text-align: center;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }
            .equation-part {
                display: inline-block;
                margin: 0.2rem 0.4rem;
                padding: 0.1rem 0.2rem;
                white-space: nowrap;
            }
            .equation-part:first-child {
                font-weight: bold;
            }
            @media (max-width: 768px) {
                .equation-container {
                    font-size: 1rem;
                    padding: 0.8rem;
                }
                .equation-part {
                    margin: 0.1rem 0.2rem;
                }
            }
            </style>
            """
            
            st.markdown(equation_css, unsafe_allow_html=True)
            st.markdown(equation_html, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📍 Intercepto (β₀)")

            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                st.metric(
                    "Valor do Intercepto (β₀)", 
                    f"{intercept:.4f}",
                    help="Valor da variável dependente quando todas as independentes são zero"
                )
                st.info(f"**Interpretação:** Quando todas as variáveis independentes são zero, o valor previsto de **{model_data['target']}** é **{intercept:.4f}**.")

            st.markdown("---")
            st.subheader("📊 Coeficientes (β₁, β₂, ..., βₙ) e Seus Impactos")

            coef_df = pd.DataFrame({
                'Variável (xᵢ)': model_data['features'],
                'Coeficiente (βᵢ)': coefficients,
                'Impacto Absoluto': np.abs(coefficients),
                'Direção': ['Positivo ↗️' if c > 0 else 'Negativo ↘️' for c in coefficients]
            }).sort_values('Impacto Absoluto', ascending=False)

            st.dataframe(coef_df.style.format({
                'Coeficiente (βᵢ)': '{:.6f}',
                'Impacto Absoluto': '{:.6f}'
            }), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("💡 Interpretação dos Coeficientes")

            st.write("**Como quantificar o impacto de cada variável:**")

            for feature, coef in zip(model_data['features'], coefficients):
                if coef > 0:
                    st.write(f"- **{feature}** (β = {coef:.6f}): A cada aumento de **1 unidade** em {feature}, "
                            f"{model_data['target']} **aumenta** em **{coef:.4f} unidades** (mantendo as demais variáveis constantes)")
                else:
                    st.write(f"- **{feature}** (β = {coef:.6f}): A cada aumento de **1 unidade** em {feature}, "
                            f"{model_data['target']} **diminui** em **{abs(coef):.4f} unidades** (mantendo as demais variáveis constantes)")

            st.markdown("---")
            st.subheader("📊 Importância Relativa das Variáveis")

            fig = go.Figure(go.Bar(
                y=coef_df['Variável (xᵢ)'],
                x=coef_df['Impacto Absoluto'],
                orientation='h',
                marker=dict(
                    color=coefficients,
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title="Coeficiente")
                ),
                text=[f"β = {c:.4f}" for c in coefficients],
                textposition='auto'
            ))

            fig.update_layout(
                xaxis_title="Magnitude do Impacto (|β|)",
                yaxis_title="Variável",
                height=max(400, len(model_data['features']) * 50),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Vermelho**: impacto positivo | **Azul**: impacto negativo. Quanto maior a barra, maior o impacto.")

        with tab2:
            st.subheader("📈 Métricas de Desempenho do Modelo")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "R² (Treino)",
                    f"{model_data['r2_train']:.4f}",
                    help="Coeficiente de Determinação (Treino) - Proporção da variância explicada"
                )

            with col2:
                st.metric(
                    "R² (Teste)",
                    f"{model_data['r2_test']:.4f}",
                    help="Coeficiente de Determinação (Teste) - Indica qualidade da generalização"
                )

            with col3:
                st.metric(
                    "MSE (Teste)",
                    f"{model_data['mse_test']:.2f}",
                    help="Erro Quadrático Médio (Teste)"
                )

            with col4:
                st.metric(
                    "RMSE (Teste)",
                    f"{model_data['rmse_test']:.2f}",
                    help="Raiz do Erro Quadrático Médio (Teste)"
                )

            st.markdown("---")
            st.subheader("📖 Interpretação do R² (Coeficiente de Determinação)")

            r2_pct = model_data['r2_test'] * 100

            if model_data['r2_test'] >= 0.8:
                st.success(f"✅ **Excelente!** O modelo explica **{r2_pct:.2f}%** da variação em {model_data['target']}. "
                          f"Isso indica que as variáveis independentes selecionadas têm alta capacidade preditiva.")
            elif model_data['r2_test'] >= 0.6:
                st.info(f"ℹ️ **Bom!** O modelo explica **{r2_pct:.2f}%** da variação em {model_data['target']}. "
                       f"As variáveis independentes têm boa capacidade preditiva.")
            elif model_data['r2_test'] >= 0.4:
                st.warning(f"⚠️ **Moderado.** O modelo explica **{r2_pct:.2f}%** da variação em {model_data['target']}. "
                          f"Considere adicionar mais variáveis ou verificar a qualidade dos dados.")
            else:
                st.error(f"❌ **Baixo.** O modelo explica apenas **{r2_pct:.2f}%** da variação em {model_data['target']}. "
                        f"O modelo precisa de melhorias significativas.")

            st.markdown("---")
            st.subheader("📊 Informações da Divisão dos Dados")

            col1, col2, col3 = st.columns(3)
            col1.metric("Amostras de Treino", len(model_data['X_train']))
            col2.metric("Amostras de Teste", len(model_data['X_test']))
            col3.metric("Total de Amostras", len(model_data['X_train']) + len(model_data['X_test']))

        with tab3:
            st.subheader("🔮 Fazer Previsões com Novos Valores")

            st.write(f"Insira os valores para as variáveis independentes e o modelo preverá o valor de **{model_data['target']}**.")

            st.markdown("---")

            input_values = {}

            num_cols = min(3, len(model_data['features']))
            cols = st.columns(num_cols)

            for idx, feature in enumerate(model_data['features']):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    min_val = float(games_df[feature].min())
                    max_val = float(games_df[feature].max())
                    mean_val = float(games_df[feature].mean())

                    input_values[feature] = st.number_input(
                        f"**{feature}**",
                        min_value=min_val * 0.5,
                        max_value=max_val * 1.5,
                        value=mean_val,
                        step=(max_val - min_val) / 100,
                        format="%.4f",
                        help=f"Média: {mean_val:.2f} | Min: {min_val:.2f} | Max: {max_val:.2f}",
                        key=f"notebook_pred_{feature}"
                    )

            if st.button("🎯 Calcular Previsão", type="primary", use_container_width=True, key="notebook_predict_button"):
                input_df = pd.DataFrame([input_values])

                prediction = model.predict(input_df)[0]

                st.markdown("---")
                st.subheader("📊 Resultado da Previsão")

                st.success(f"### 🎯 Valor Previsto de {model_data['target']}: **{prediction:.2f}**")

                st.markdown("---")
                st.subheader("🧮 Cálculo Detalhado (Equação 1)")

                calc_html = f"""
                <div class="calculation-container">
                    <span class="calc-part"><strong>{model_data['target']}</strong></span>
                """
                
                if model.intercept_ >= 0:
                    calc_html += f'<span class="calc-part">= {model.intercept_:.4f}</span>'
                else:
                    calc_html += f'<span class="calc-part">= {model.intercept_:.4f}</span>'
                
                for feature, coef in zip(model_data['features'], model.coef_):
                    value = input_values[feature]
                    sign = "+" if coef >= 0 else ""
                    term_html = f'<span class="calc-part">{sign} ({coef:.4f} × {value:.4f})</span>'
                    calc_html += term_html
                
                calc_html += f'<br><span class="calc-result">= {prediction:.4f}</span></div>'
                
                calc_css = """
                <style>
                .calculation-container {
                    font-family: 'Courier New', monospace;
                    font-size: 1.1em;
                    line-height: 1.8;
                    padding: 1rem;
                    margin: 1rem 0;
                    text-align: center;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                }
                .calc-part {
                    display: inline-block;
                    margin: 0.2rem 0.3rem;
                    padding: 0.1rem 0.2rem;
                    white-space: nowrap;
                }
                .calc-result {
                    font-weight: bold;
                    font-size: 1.1em;
                    margin-top: 0.5rem;
                    display: inline-block;
                    padding: 0.2rem 0.3rem;
                }
                </style>
                """
                
                st.markdown(calc_css, unsafe_allow_html=True)
                st.markdown(calc_html, unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("📊 Contribuição de Cada Variável")

                contributions = []
                for feature, coef in zip(model_data['features'], model.coef_):
                    value = input_values[feature]
                    contribution = coef * value
                    contributions.append({
                        'Variável': feature,
                        'Valor Inserido (x)': value,
                        'Coeficiente (β)': coef,
                        'Contribuição (β × x)': contribution,
                        'Porcentagem do Total': 0 
                    })

                intercepto_row = {
                    'Variável': 'Intercepto (β₀)',
                    'Valor Inserido (x)': 1.0,
                    'Coeficiente (β)': model.intercept_,
                    'Contribuição (β × x)': model.intercept_,
                    'Porcentagem do Total': 0
                }

                contrib_df = pd.DataFrame([intercepto_row] + contributions)

                total_positive = contrib_df[contrib_df['Contribuição (β × x)'] > 0]['Contribuição (β × x)'].sum()
                if total_positive > 0:
                    contrib_df['Porcentagem do Total'] = (contrib_df['Contribuição (β × x)'] / total_positive * 100).clip(lower=0)

                st.dataframe(contrib_df.style.format({
                    'Valor Inserido (x)': '{:.4f}',
                    'Coeficiente (β)': '{:.6f}',
                    'Contribuição (β × x)': '{:.4f}',
                    'Porcentagem do Total': '{:.2f}%'
                }), use_container_width=True, hide_index=True)

                total_contribution = model.intercept_ + sum([c['Contribuição (β × x)'] for c in contributions])
                st.info(f"**✅ Soma Total das Contribuições = {total_contribution:.4f}** ≈ **{prediction:.4f}** (valor previsto)")

                st.markdown("---")
                st.subheader("📊 Visualização das Contribuições")

                fig = go.Figure(go.Bar(
                    x=contrib_df['Contribuição (β × x)'],
                    y=contrib_df['Variável'],
                    orientation='h',
                    marker=dict(
                        color=contrib_df['Contribuição (β × x)'],
                        colorscale='RdBu',
                        showscale=True,
                        colorbar=dict(title="Contribuição")
                    ),
                    text=contrib_df['Contribuição (β × x)'].apply(lambda x: f"{x:.2f}"),
                    textposition='auto'
                ))

                fig.update_layout(
                    xaxis_title="Contribuição para a Previsão",
                    yaxis_title="Variável",
                    height=max(400, len(model_data['features']) * 50)
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Vermelho**: contribuição positiva (aumenta o valor previsto) | **Azul**: contribuição negativa (diminui o valor previsto)")

        with tab4:
            st.subheader("📊 Visualizações da Regressão")

            st.markdown("### 1. Valores Reais vs Valores Preditos")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=model_data['y_train'].values,
                y=model_data['y_pred_train'],
                mode='markers',
                name='Treino',
                marker=dict(color='blue', size=8, opacity=0.6),
                text=[f"Real: {r:.2f}<br>Predito: {p:.2f}"
                      for r, p in zip(model_data['y_train'].values, model_data['y_pred_train'])],
                hovertemplate='%{text}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=model_data['y_test'].values,
                y=model_data['y_pred_test'],
                mode='markers',
                name='Teste',
                marker=dict(color='red', size=10, opacity=0.8),
                text=[f"Real: {r:.2f}<br>Predito: {p:.2f}"
                      for r, p in zip(model_data['y_test'].values, model_data['y_pred_test'])],
                hovertemplate='%{text}<extra></extra>'
            ))

            all_values = np.concatenate([model_data['y_train'].values, model_data['y_test'].values,
                                         model_data['y_pred_train'], model_data['y_pred_test']])
            min_val = all_values.min()
            max_val = all_values.max()

            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predição Perfeita',
                line=dict(color='green', dash='dash', width=2)
            ))

            fig.update_layout(
                xaxis_title=f"Valores Reais de {model_data['target']}",
                yaxis_title=f"Valores Preditos de {model_data['target']}",
                height=500,
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 Quanto mais próximos da linha verde (diagonal perfeita), melhor é a predição do modelo.")

            st.markdown("---")
            st.markdown("### 2. Análise de Resíduos")

            residuals_train = model_data['y_train'].values - model_data['y_pred_train']
            residuals_test = model_data['y_test'].values - model_data['y_pred_test']

            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=model_data['y_pred_train'],
                y=residuals_train,
                mode='markers',
                name='Treino',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))

            fig2.add_trace(go.Scatter(
                x=model_data['y_pred_test'],
                y=residuals_test,
                mode='markers',
                name='Teste',
                marker=dict(color='red', size=10, opacity=0.8)
            ))

            fig2.add_hline(y=0, line_dash="dash", line_color="green", line_width=2)

            fig2.update_layout(
                xaxis_title=f"Valores Preditos de {model_data['target']}",
                yaxis_title="Resíduos (Real - Predito)",
                height=500
            )

            st.plotly_chart(fig2, use_container_width=True)
            st.caption("📌 Os resíduos devem estar distribuídos aleatoriamente em torno de zero. Padrões podem indicar problemas no modelo.")

            if len(model_data['features']) > 0:
                st.markdown("---")
                st.markdown(f"### 3. Linha de Regressão Ajustada - {model_data['features'][0]}")

                first_feature = model_data['features'][0]

                X_simple = games_df[[first_feature]]
                y_simple = games_df[model_data['target']]

                simple_model = LinearRegression()
                simple_model.fit(X_simple, y_simple)

                x_range = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
                y_range = simple_model.predict(x_range)

                fig3 = go.Figure()

                fig3.add_trace(go.Scatter(
                    x=games_df[first_feature],
                    y=games_df[model_data['target']],
                    mode='markers',
                    name='Dados Reais',
                    marker=dict(color='blue', size=8, opacity=0.6)
                ))

                fig3.add_trace(go.Scatter(
                    x=x_range.flatten(),
                    y=y_range,
                    mode='lines',
                    name='Linha de Regressão',
                    line=dict(color='red', width=3)
                ))

                fig3.update_layout(
                    xaxis_title=first_feature,
                    yaxis_title=model_data['target'],
                    height=500,
                    title=f"Melhor Linha Reta Ajustada: {model_data['target']} vs {first_feature}"
                )

                st.plotly_chart(fig3, use_container_width=True)
                st.caption(f"📌 A linha vermelha representa a melhor linha reta que se ajusta aos dados (minimiza o erro quadrático).")
                st.info(f"**Equação da linha:** {model_data['target']} = {simple_model.intercept_:.4f} + {simple_model.coef_[0]:.4f} × {first_feature}")

def logistic_regression_theory_view(games_df):
    """Interface interativa de Regressão Logística"""
    st.header("📊 Regressão Logística - Predição com Probabilidades")

    df = games_df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    cols_to_exclude = ['data-jogo']
    available_cols = [col for col in numeric_cols if col not in cols_to_exclude]

    st.markdown("### 🎯 Configuração do Modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Variável Dependente (Y)")
        st.info("""
        **Escolha a variável que deseja prever.**

        Para variáveis contínuas, será criada uma classificação binária baseada em um limiar.
        """)

        target_options = {
            'resultado': 'Resultado do Jogo (Vitória/Derrota)',
            'pontos': 'Pontos (Alto/Baixo)',
            'assistencias': 'Assistências (Alto/Baixo)',
            'rebotes-totais': 'Rebotes Totais (Alto/Baixo)',
            'triplos-convertidos': 'Triplos Convertidos (Alto/Baixo)',
            'porcentagem-arremessos': 'Porcentagem de Arremessos (Alto/Baixo)'
        }

        available_targets = {k: v for k, v in target_options.items() if k in available_cols}

        target_var = st.selectbox(
            "Selecione a Variável Y:",
            options=list(available_targets.keys()),
            format_func=lambda x: available_targets[x],
            key="logistic_target_var"
        )

        if target_var != 'resultado':
            min_val = float(df[target_var].min())
            max_val = float(df[target_var].max())
            mean_val = float(df[target_var].mean())

            threshold = st.slider(
                f"Limiar para classificação de {target_options[target_var]}:",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                help=f"Valores acima do limiar serão classificados como 1 (Alto), abaixo como 0 (Baixo)",
                key=f"logistic_threshold_{target_var}"
            )
            df['target'] = (df[target_var] > threshold).astype(int)
        else:
            df['target'] = df['resultado']

    with col2:
        st.markdown("#### Variáveis Independentes (X)")
        st.info("""
        **Escolha as variáveis que deseja usar para fazer a predição.**

        Selecione uma ou mais variáveis da base de dados.
        """)

        available_features = [col for col in available_cols if col != target_var]

        feature_labels = {
            'arremessos-tentados': 'Arremessos Tentados',
            'arremessos-convertidos': 'Arremessos Convertidos',
            'porcentagem-arremessos': 'Porcentagem de Arremessos',
            'triplos-tentados': 'Triplos Tentados',
            'triplos-convertidos': 'Triplos Convertidos',
            'porcentagem-triplos': 'Porcentagem de Triplos',
            'lances-livres-tentados': 'Lances Livres Tentados',
            'lances-livres-convertidos': 'Lances Livres Convertidos',
            'porcentagem-lances-livres': 'Porcentagem de Lances Livres',
            'rebotes-totais': 'Rebotes Totais',
            'rebotes-ofensivos': 'Rebotes Ofensivos',
            'rebotes-defensivos': 'Rebotes Defensivos',
            'assistencias': 'Assistências',
            'roubos': 'Roubos de Bola',
            'tocos': 'Tocos',
            'erros': 'Erros',
            'faltas': 'Faltas',
            'pontos': 'Pontos',
            'saldo-pontos': 'Saldo de Pontos',
            'mando-de-jogo': 'Mando de Jogo (Casa/Fora)'
        }

        selected_features = st.multiselect(
            "Selecione as Variáveis X:",
            options=available_features,
            default=available_features[:4] if len(available_features) >= 4 else available_features,
            format_func=lambda x: feature_labels.get(x, x),
            key="logistic_selected_features"
        )

    if not selected_features:
        st.warning("⚠️ Selecione pelo menos uma variável independente (X) para treinar o modelo.")
        return

    st.markdown("---")

    try:
        if 'target' not in df.columns:
            if target_var == 'resultado':
                df['target'] = df['resultado']
            else:
                min_val = float(df[target_var].min())
                max_val = float(df[target_var].max())
                mean_val = float(df[target_var].mean())
                threshold = mean_val  # usar média como padrão
                df['target'] = (df[target_var] > threshold).astype(int)
        
        st.info(f"🔍 Debug: Variável target criada com {df['target'].sum()} valores classe 1 de {len(df)} total")
        
        X = df[selected_features].dropna()
        y = df.loc[X.index, 'target']

        if len(X) < 10:
            st.warning("⚠️ Dados insuficientes para treinar o modelo. Selecione outras variáveis.")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.markdown("### 📊 Resultados do Modelo")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Acurácia", f"{accuracy:.1%}")

        with col2:
            correct = cm[0][0] + cm[1][1] if len(cm) > 1 else cm[0][0]
            total = len(y_test)
            st.metric("Predições Corretas", f"{correct}/{total}")

        with col3:
            class_1_count = y.sum()
            st.metric("Classe 1 (Alto/Vitória)", f"{class_1_count}")

        with col4:
            class_0_count = len(y) - class_1_count
            st.metric("Classe 0 (Baixo/Derrota)", f"{class_0_count}")

        st.markdown("### 🔢 Equação do Modelo")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"**Intercepto (β₀):** {model.intercept_[0]:.4f}")

            coef_df = pd.DataFrame({
                'Variável': [feature_labels.get(f, f) for f in selected_features],
                'Coeficiente': model.coef_[0]
            }).sort_values('Coeficiente', ascending=False)

            st.dataframe(coef_df, use_container_width=True, hide_index=True)

        with col2:
            fig_coef = px.bar(
                coef_df,
                x='Coeficiente',
                y='Variável',
                orientation='h',
                title='Importância das Variáveis',
                color='Coeficiente',
                color_continuous_scale='RdYlGn'
            )
            fig_coef.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_coef, use_container_width=True)

        st.markdown("**Equação Completa:**")

        equation_text = f"**z** = {model.intercept_[0]:.4f}"
        for i, feature in enumerate(selected_features):
            coef = model.coef_[0][i]
            sign = "+" if coef >= 0 else ""
            feature_name = feature_labels.get(feature, feature)
            equation_text += f" {sign} {abs(coef):.4f} × {feature_name}"

        st.markdown(equation_text)
        st.markdown("**p(Classe 1)** = 1 / [1 + e^(-z)]")

        st.markdown("---")
        st.markdown("**📝 Exemplo de Cálculo:**")

        if len(X_test) > 0:
            example_idx = 0
            example_values = X.iloc[example_idx]
            example_scaled = scaler.transform([example_values.values])[0]

            z_value = model.intercept_[0]
            for i, feature in enumerate(selected_features):
                z_value += model.coef_[0][i] * example_scaled[i]

            p_value = 1 / (1 + np.exp(-z_value))

            st.markdown("Para os seguintes valores:")
            values_text = ""
            for feature in selected_features:
                feature_name = feature_labels.get(feature, feature)
                values_text += f"- **{feature_name}**: {example_values[feature]:.2f}\n"
            st.markdown(values_text)

            st.markdown(f"Calculamos **z** = {z_value:.4f}")
            st.markdown(f"E então **p(Classe 1)** = 1 / [1 + e^(-{z_value:.4f})] = **{p_value:.4f}** ({p_value*100:.2f}%)")

            if p_value > 0.5:
                st.success(f"✅ Neste exemplo, o modelo prevê **Classe 1** (probabilidade > 50%)")
            else:
                st.info(f"ℹ️ Neste exemplo, o modelo prevê **Classe 0** (probabilidade < 50%)")

        st.markdown("---")
        st.markdown("### 🎯 Fazer Predição Personalizada")

        st.info("Insira os valores das variáveis para calcular a probabilidade:")

        prediction_values = {}

        cols = st.columns(3)
        for i, feature in enumerate(selected_features):
            with cols[i % 3]:
                mean_val = float(X[feature].mean())
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())

                prediction_values[feature] = st.number_input(
                    feature_labels.get(feature, feature),
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"logistic_pred_{feature}",
                    help=f"Média: {mean_val:.2f}"
                )

        if st.button("🔮 Calcular Probabilidade", type="primary", key="logistic_calculate_probability"):
            pred_input = np.array([list(prediction_values.values())])
            pred_input_scaled = scaler.transform(pred_input)

            prediction = model.predict(pred_input_scaled)[0]
            probability = model.predict_proba(pred_input_scaled)[0]

            prob_class_0 = probability[0]
            prob_class_1 = probability[1]

            st.markdown("---")
            st.markdown("### 📈 Resultado da Predição")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Probabilidade Classe 0 (Baixo/Derrota)",
                    f"{prob_class_0:.1%}"
                )

            with col2:
                st.metric(
                    "Probabilidade Classe 1 (Alto/Vitória)",
                    f"{prob_class_1:.1%}"
                )

            with col3:
                result_text = "Classe 1 (Alto/Vitória)" if prediction == 1 else "Classe 0 (Baixo/Derrota)"
                st.metric("Predição Final", result_text)

            if prob_class_1 > 0.5:
                st.success(f"✅ O modelo prevê **Classe 1** com {prob_class_1:.1%} de probabilidade")
            else:
                st.info(f"ℹ️ O modelo prevê **Classe 0** com {prob_class_0:.1%} de probabilidade")

        st.markdown("---")
        st.markdown("### 📊 Visualizações")

        tab1, tab2 = st.tabs(["Matriz de Confusão", "Distribuição de Probabilidades"])

        with tab1:
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predito", y="Real", color="Contagem"),
                x=['Classe 0', 'Classe 1'],
                y=['Classe 0', 'Classe 1'],
                text_auto=True,
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        with tab2:
            prob_df = pd.DataFrame({
                'Probabilidade Classe 1': y_pred_proba[:, 1],
                'Classe Real': y_test.map({0: 'Classe 0', 1: 'Classe 1'})
            })

            fig_prob = px.histogram(
                prob_df,
                x='Probabilidade Classe 1',
                color='Classe Real',
                nbins=20,
                title='Distribuição de Probabilidades por Classe Real',
                labels={'Probabilidade Classe 1': 'Probabilidade de Classe 1'},
                barmode='overlay',
                opacity=0.7
            )
            fig_prob.update_layout(height=400)
            st.plotly_chart(fig_prob, use_container_width=True)

    except KeyError as ke:
        st.error(f"Erro: Coluna não encontrada - {ke}")
        st.info("Verifique se todas as variáveis selecionadas existem nos dados.")
    except ValueError as ve:
        st.error(f"Erro de valor: {ve}")
        st.info("Verifique se os dados são válidos para o modelo.")
    except Exception as e:
        st.error(f"Erro inesperado ao treinar o modelo: {e}")
        st.info("Tente selecionar outras variáveis ou verifique os dados.")


def main():
    """Função principal da aplicação"""
    st.title("🏀 Dallas Mavericks 2024-25")
    st.markdown("### Análise Exploratória de Dados - Temporada 2024-25")

    players_df, games_df = load_data()

    if players_df is None or games_df is None:
        st.error("Não foi possível carregar os dados. Verifique se os arquivos estão no local correto.")
        return

    st.sidebar.header("🎛️ Filtros Globais")
    
    min_minutes = st.sidebar.slider(
        "Minutos mínimos totais",
        min_value=0,
        max_value=int(players_df['minutos_total'].max()),
        value=0,
        step=50,
        help="Filtrar jogadores por minutos mínimos jogados na temporada"
    )
    
    positions = sorted(players_df['posicao-g-f-fc-cf-c'].unique())
    position_names = {1: 'G', 2: 'F', 3: 'FC', 4: 'CF', 5: 'C'}
    selected_positions = st.sidebar.multiselect(
        "Posições", 
        options=positions,
        default=positions,
        format_func=lambda x: position_names.get(x, f"Posição {x}"),
        help="Selecionar posições para análise"
    )
    
    filtered_players = players_df[
        (players_df['minutos_total'] >= min_minutes) &
        (players_df['posicao-g-f-fc-cf-c'].isin(selected_positions))
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Jogadores selecionados:** {len(filtered_players)}/{len(players_df)}")

    if len(filtered_players) == 0:
        st.warning("⚠️ Nenhum jogador atende aos critérios selecionados. Ajuste os filtros na sidebar.")
        return

    create_summary_metrics(filtered_players, games_df)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "👥 Jogadores",
        "🏀 Jogos",
        "🧠 Análise Avançada",
        "🔍 Interativa",
        "🎯 Predições Específicas",
        "📈 Regressão Linear",
        "📊 Regressão Logística"
    ])

    with tab1:
        player_analysis(filtered_players)

    with tab2:
        game_analysis(games_df)

    with tab3:
        advanced_analysis(filtered_players, games_df)

    with tab4:
        interactive_analysis(filtered_players, games_df)

    with tab5:
        prediction_interface(filtered_players, games_df)

    with tab6:
        notebook_regression_analysis(games_df)

    with tab7:
        logistic_regression_theory_view(games_df)
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.8em;'>
            <p>📊 Dashboard desenvolvido para análise exploratória dos dados dos Dallas Mavericks</p>
            <p>Temporada 2024-25 • Dados processados e limpos automaticamente</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
