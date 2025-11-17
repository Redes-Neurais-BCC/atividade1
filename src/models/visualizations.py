"""
Modulo de Visualizacoes para Redes Neurais
==========================================
Graficos e visualizacoes para analise de modelos de Machine Learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy import stats


class ModelVisualizer:
    """
    Classe para criar visualizacoes de modelos de ML.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Inicializa o visualizador.

        Args:
            style: Estilo dos graficos matplotlib
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')

        sns.set_palette("husl")
        self.colors = {
            'train': '#2ecc71',
            'val': '#e74c3c',
            'test': '#3498db',
            'pred': '#9b59b6',
            'actual': '#f39c12'
        }

    def plot_training_evolution(
        self,
        history: Dict,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plota a evolucao do erro durante o treinamento.

        Mostra:
        - Erro (loss) nos dados de treino
        - Erro (loss) nos dados de validacao
        - Accuracy nos dados de treino
        - Accuracy nos dados de validacao

        Args:
            history: Dicionario com historico de treinamento
            save_path: Caminho para salvar a figura
            show: Se True, exibe o grafico

        Returns:
            Figura matplotlib
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Extrair dados
        if 'loss' in history:
            train_loss = history['loss']
            val_loss = history.get('val_loss', None)
            train_acc = history.get('accuracy', history.get('acc', None))
            val_acc = history.get('val_accuracy', history.get('val_acc', None))
        else:
            # Formato alternativo (MLPTrainer)
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            train_acc = history.get('train_accuracy', [])
            val_acc = history.get('val_accuracy', [])

        epochs = range(1, len(train_loss) + 1)

        # Plot 1: Loss
        axes[0].plot(epochs, train_loss, 'o-',
                    color=self.colors['train'], label='Treino', linewidth=2, markersize=4)
        if val_loss:
            axes[0].plot(epochs, val_loss, 's-',
                        color=self.colors['val'], label='Validacao', linewidth=2, markersize=4)

        axes[0].set_xlabel('Epoca', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss (Erro)', fontsize=12, fontweight='bold')
        axes[0].set_title('Evolucao do Erro Durante Treinamento', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Adicionar anotacoes
        if val_loss:
            # Identificar overfitting
            if len(val_loss) > 10:
                # Se validacao comecar a subir enquanto treino desce
                mid_point = len(train_loss) // 2
                train_trend = train_loss[mid_point] - train_loss[-1]
                val_trend = val_loss[-1] - val_loss[mid_point]

                if train_trend > 0 and val_trend > 0:
                    axes[0].text(0.5, 0.95, 'ALERTA: Possivel Overfitting',
                               transform=axes[0].transAxes,
                               fontsize=10, color='red',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                               verticalalignment='top',
                               horizontalalignment='center')

        # Plot 2: Accuracy
        if train_acc:
            axes[1].plot(epochs, train_acc, 'o-',
                        color=self.colors['train'], label='Treino', linewidth=2, markersize=4)
            if val_acc:
                axes[1].plot(epochs, val_acc, 's-',
                            color=self.colors['val'], label='Validacao', linewidth=2, markersize=4)

            axes[1].set_xlabel('Epoca', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Acuracia', fontsize=12, fontweight='bold')
            axes[1].set_title('Evolucao da Acuracia Durante Treinamento', fontsize=14, fontweight='bold')
            axes[1].legend(loc='lower right', fontsize=10)
            axes[1].grid(True, alpha=0.3)

            # Linha de convergencia
            if len(train_acc) > 20:
                recent_std = np.std(train_acc[-10:])
                if recent_std < 0.01:
                    axes[1].text(0.5, 0.05, 'Modelo Convergiu',
                               transform=axes[1].transAxes,
                               fontsize=10, color='green',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                               verticalalignment='bottom',
                               horizontalalignment='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Grafico salvo em: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_error_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plota matriz de erros com 3 graficos:
        1. Histograma dos erros
        2. Scatter plot: Valores Previstos vs Reais
        3. Scatter plot: Residuos vs Valores Previstos

        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            save_path: Caminho para salvar a figura
            show: Se True, exibe o grafico

        Returns:
            Figura matplotlib
        """
        fig = plt.figure(figsize=(18, 5))

        # Calcular erros e residuos
        errors = y_pred - y_true
        residuals = errors

        # Plot 1: Histograma dos erros
        ax1 = plt.subplot(1, 3, 1)
        ax1.hist(errors, bins=30, color=self.colors['pred'],
                alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
        ax1.axvline(x=np.mean(errors), color='green', linestyle='--',
                   linewidth=2, label=f'Media: {np.mean(errors):.3f}')

        ax1.set_xlabel('Erro (Previsto - Real)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequencia', fontsize=12, fontweight='bold')
        ax1.set_title('Distribuicao dos Erros', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Adicionar estatisticas
        textstr = f'Media: {np.mean(errors):.3f}\nStd: {np.std(errors):.3f}\nMAE: {np.mean(np.abs(errors)):.3f}'
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Scatter - Previsto vs Real
        ax2 = plt.subplot(1, 3, 2)
        ax2.scatter(y_true, y_pred, alpha=0.6, color=self.colors['pred'], s=50)

        # Linha de perfeita predicao
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Predicao Perfeita')

        # Linha de regressao
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax2.plot(y_true, p(y_true), 'g-', linewidth=2,
                label=f'Regressao: y={z[0]:.2f}x+{z[1]:.2f}')

        ax2.set_xlabel('Valores Reais', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Valores Previstos', fontsize=12, fontweight='bold')
        ax2.set_title('Previsao vs Realidade', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # R-squared
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        r_squared = correlation ** 2
        ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}',
                transform=ax2.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Plot 3: Scatter - Residuos vs Previsto
        ax3 = plt.subplot(1, 3, 3)
        ax3.scatter(y_pred, residuals, alpha=0.6, color=self.colors['actual'], s=50)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)

        # Adicionar bandas de confianca
        std_residuals = np.std(residuals)
        ax3.axhline(y=2*std_residuals, color='orange', linestyle=':',
                   linewidth=1.5, label='+2 Std')
        ax3.axhline(y=-2*std_residuals, color='orange', linestyle=':',
                   linewidth=1.5, label='-2 Std')

        ax3.set_xlabel('Valores Previstos', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Residuos', fontsize=12, fontweight='bold')
        ax3.set_title('Analise de Residuos', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Grafico salvo em: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_prediction_ranking(
        self,
        data: pd.DataFrame,
        statistic: str,
        top_n: int = 20,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plota ranking dos jogadores com maiores previsoes para uma estatistica.

        Args:
            data: DataFrame com previsoes (deve ter colunas: 'jogador', 'previsao', 'real')
            statistic: Nome da estatistica
            top_n: Numero de jogadores a mostrar
            save_path: Caminho para salvar a figura
            show: Se True, exibe o grafico

        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        # Ordenar por previsao
        data_sorted = data.nlargest(top_n, 'previsao')

        # Criar posicoes
        y_pos = np.arange(len(data_sorted))

        # Plot barras
        bars1 = ax.barh(y_pos - 0.2, data_sorted['previsao'], 0.4,
                       label='Previsao', color=self.colors['pred'])
        bars2 = ax.barh(y_pos + 0.2, data_sorted['real'], 0.4,
                       label='Real', color=self.colors['actual'])

        # Customizacao
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data_sorted['jogador'])
        ax.invert_yaxis()
        ax.set_xlabel(f'{statistic}', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} - Ranking de Previsoes: {statistic}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        # Adicionar valores nas barras
        for i, (pred, real) in enumerate(zip(data_sorted['previsao'], data_sorted['real'])):
            ax.text(pred, i - 0.2, f' {pred:.1f}', va='center', fontsize=8)
            ax.text(real, i + 0.2, f' {real:.1f}', va='center', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Grafico salvo em: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_home_away_comparison(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plota comparacao de performance entre jogos em casa vs fora.

        Args:
            data: DataFrame com dados (deve ter coluna 'home_away' e metricas)
            metrics: Lista de metricas a comparar
            save_path: Caminho para salvar a figura
            show: Se True, exibe o grafico

        Returns:
            Figura matplotlib
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))

        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            # Separar dados
            home_data = data[data['home_away'] == 'home'][metric]
            away_data = data[data['home_away'] == 'away'][metric]

            # Box plot
            bp = axes[idx].boxplot([home_data, away_data],
                                   labels=['Casa', 'Fora'],
                                   patch_artist=True,
                                   showmeans=True)

            # Colorir boxes
            colors = [self.colors['train'], self.colors['val']]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[idx].set_ylabel(metric, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{metric}: Casa vs Fora', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

            # Adicionar estatisticas
            home_mean = home_data.mean()
            away_mean = away_data.mean()
            diff = ((home_mean - away_mean) / away_mean) * 100

            textstr = f'Casa: {home_mean:.2f}\nFora: {away_mean:.2f}\nDiff: {diff:+.1f}%'
            axes[idx].text(0.05, 0.95, textstr, transform=axes[idx].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Grafico salvo em: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_temporal_evolution(
        self,
        data: pd.DataFrame,
        date_column: str,
        value_column: str,
        prediction_column: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plota evolucao temporal de uma metrica.

        Args:
            data: DataFrame com dados
            date_column: Nome da coluna de data
            value_column: Nome da coluna de valores reais
            prediction_column: Nome da coluna de previsoes (opcional)
            save_path: Caminho para salvar a figura
            show: Se True, exibe o grafico

        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=(15, 6))

        # Converter datas
        data_copy = data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column])
        data_copy = data_copy.sort_values(date_column)

        # Plot valores reais
        ax.plot(data_copy[date_column], data_copy[value_column],
               'o-', color=self.colors['actual'], label='Real',
               linewidth=2, markersize=6, alpha=0.7)

        # Plot previsoes se fornecidas
        if prediction_column and prediction_column in data_copy.columns:
            ax.plot(data_copy[date_column], data_copy[prediction_column],
                   's--', color=self.colors['pred'], label='Previsao',
                   linewidth=2, markersize=6, alpha=0.7)

        # Adicionar media movel
        window = min(5, len(data_copy) // 4)
        if window >= 2:
            rolling_mean = data_copy[value_column].rolling(window=window).mean()
            ax.plot(data_copy[date_column], rolling_mean,
                   '-', color='green', label=f'Media Movel ({window} jogos)',
                   linewidth=2.5, alpha=0.8)

        ax.set_xlabel('Data', fontsize=12, fontweight='bold')
        ax.set_ylabel(value_column, fontsize=12, fontweight='bold')
        ax.set_title(f'Evolucao Temporal: {value_column}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotacionar labels de data
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Grafico salvo em: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_confidence_intervals(
        self,
        predictions: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Tuple[plt.Figure, Dict]:
        """
        Plota intervalos de confianca para as previsoes usando bootstrap.

        Args:
            predictions: Array de previsoes
            confidence_level: Nivel de confianca (default: 0.95 = 95%)
            n_bootstrap: Numero de amostras bootstrap
            save_path: Caminho para salvar a figura
            show: Se True, exibe o grafico

        Returns:
            Tupla (figura, dicionario com intervalos)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Bootstrap
        bootstrap_means = []
        n_samples = len(predictions)

        for _ in range(n_bootstrap):
            sample = np.random.choice(predictions, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        # Calcular intervalos
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        mean_pred = np.mean(predictions)

        # Plot histograma do bootstrap
        ax.hist(bootstrap_means, bins=50, color=self.colors['pred'],
               alpha=0.7, edgecolor='black', density=True)

        # Linhas de intervalo
        ax.axvline(mean_pred, color='red', linestyle='-',
                  linewidth=2.5, label=f'Media: {mean_pred:.3f}')
        ax.axvline(ci_lower, color='green', linestyle='--',
                  linewidth=2, label=f'IC {confidence_level*100:.0f}%: [{ci_lower:.3f}, {ci_upper:.3f}]')
        ax.axvline(ci_upper, color='green', linestyle='--', linewidth=2)

        # Sombrear area de confianca
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='green')

        ax.set_xlabel('Media das Previsoes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Densidade', fontsize=12, fontweight='bold')
        ax.set_title(f'Intervalo de Confianca ({confidence_level*100:.0f}%) - Bootstrap',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Adicionar estatisticas
        textstr = f'N Bootstrap: {n_bootstrap}\nMedia: {mean_pred:.3f}\nStd: {np.std(bootstrap_means):.3f}\nIC: [{ci_lower:.3f}, {ci_upper:.3f}]'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Grafico salvo em: {save_path}")

        if show:
            plt.show()

        intervals = {
            'mean': mean_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'std': np.std(bootstrap_means)
        }

        return fig, intervals

    def create_prediction_table(
        self,
        data: pd.DataFrame,
        statistics: List[str],
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Cria tabela de Previsao x Realidade similar ao screenshot.

        Args:
            data: DataFrame com dados (deve ter colunas: 'jogador' e pares 'stat_pred'/'stat_real')
            statistics: Lista de estatisticas (ex: ['pontos', 'assistencias', 'rebotes'])
            top_n: Numero de jogadores a mostrar
            save_path: Caminho para salvar CSV (opcional)

        Returns:
            DataFrame formatado
        """
        # Selecionar top jogadores por alguma metrica
        if 'pontos_pred' in data.columns:
            data_sorted = data.nlargest(top_n, 'pontos_pred')
        else:
            data_sorted = data.head(top_n)

        # Criar tabela formatada
        table_data = []

        for _, row in data_sorted.iterrows():
            player_row = {'Jogador': row['jogador']}

            for stat in statistics:
                pred_col = f'{stat}_pred'
                real_col = f'{stat}_real'

                if pred_col in row and real_col in row:
                    player_row[f'Previsto_{stat.capitalize()}'] = f"{row[pred_col]:.0f}"
                    player_row[f'Realidade_{stat.capitalize()}'] = f"{row[real_col]:.0f}"

            table_data.append(player_row)

        result_df = pd.DataFrame(table_data)

        if save_path:
            result_df.to_csv(save_path, index=False)
            print(f"[OK] Tabela salva em: {save_path}")

        return result_df

    def plot_all_diagnostics(
        self,
        history: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_dir: Optional[str] = None
    ):
        """
        Plota todos os graficos de diagnostico de uma vez.

        Args:
            history: Historico de treinamento
            y_true: Valores reais
            y_pred: Valores previstos
            save_dir: Diretorio para salvar graficos (opcional)
        """
        print("\n" + "="*70)
        print("GERANDO GRAFICOS DE DIAGNOSTICO")
        print("="*70 + "\n")

        # 1. Evolucao do erro
        print("1. Evolucao do Erro durante Treinamento...")
        save_path = f"{save_dir}/training_evolution.png" if save_dir else None
        self.plot_training_evolution(history, save_path=save_path, show=False)

        # 2. Matriz de erros
        print("2. Matriz de Erros...")
        save_path = f"{save_dir}/error_matrix.png" if save_dir else None
        self.plot_error_matrix(y_true, y_pred, save_path=save_path, show=False)

        # 3. Intervalos de confianca
        print("3. Intervalos de Confianca...")
        save_path = f"{save_dir}/confidence_intervals.png" if save_dir else None
        self.plot_confidence_intervals(y_pred, save_path=save_path, show=False)

        print("\n[OK] Todos os graficos de diagnostico foram gerados!")
        print("="*70 + "\n")
