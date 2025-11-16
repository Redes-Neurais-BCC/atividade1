import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Optional
import time

from .mlp import MLP
from .feature_selector import FeatureSelector, prepare_features_and_target
from .visualizations import ModelVisualizer


class MLPTrainer:
    """
    Pipeline completo para treinamento de MLP.

    Inclui:
    - Pré-processamento de dados
    - Normalização
    - Treinamento
    - Avaliação
    - Métricas de performance
    """

    def __init__(
        self,
        data_path: str,
        target_column: str = 'resultado',
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Inicializa o trainer.

        Args:
            data_path: Caminho para o arquivo CSV
            target_column: Nome da coluna target
            test_size: Proporção dos dados para teste
            random_state: Seed para reprodutibilidade
        """
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        # Componentes
        self.df = None
        self.feature_selector = FeatureSelector(target_column)
        self.scaler = StandardScaler()
        self.mlp = None
        self.visualizer = ModelVisualizer()

        # Dados
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

        # Resultados
        self.training_time = None
        self.predictions = None
        self.metrics = None

    def load_data(self) -> pd.DataFrame:
        """Carrega os dados do CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"[OK] Dados carregados: {self.df.shape[0]} amostras, {self.df.shape[1]} colunas")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {self.data_path}")

    def select_features(
        self,
        method: str = 'correlation',
        n_features: int = 6,
        custom_features: Optional[List[str]] = None
    ) -> List[str]:
        if self.df is None:
            raise ValueError("Carregue os dados primeiro usando load_data()")

        if method == 'correlation':
            features = self.feature_selector.select_by_correlation(
                self.df,
                n_features=n_features
            )
            print(f"\n[OK] Selecionadas {len(features)} features por correlacao:")
            for i, f in enumerate(features, 1):
                corr = self.feature_selector.correlation_scores[f]
                print(f"  {i}. {f}: {corr:.4f}")

        elif method == 'all':
            features = self.feature_selector.get_all_features(self.df)
            print(f"\n[OK] Usando todas as {len(features)} features disponiveis")

        elif method == 'predefined':
            features = self.feature_selector.get_predefined_features('logistic')
            print(f"\n[OK] Usando {len(features)} features predefinidas (modelo logistico)")

        elif method == 'custom' and custom_features:
            features = custom_features
            print(f"\n[OK] Usando {len(features)} features customizadas")

        else:
            raise ValueError(f"Método '{method}' inválido ou features não fornecidas")

        return features

    def prepare_data(self, features: List[str], normalize: bool = True):
        # Preparar X e y
        X, y = prepare_features_and_target(self.df, features, self.target_column)

        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Manter proporção de classes
        )

        # Normalizar
        if normalize:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test

        print(f"\n[OK] Dados preparados:")
        print(f"  - Treino: {self.X_train.shape[0]} amostras")
        print(f"  - Teste:  {self.X_test.shape[0]} amostras")
        print(f"  - Features: {len(features)}")
        print(f"  - Normalizacao: {'Sim' if normalize else 'Nao'}")

        # Distribuicao de classes
        train_dist = pd.Series(self.y_train).value_counts()
        test_dist = pd.Series(self.y_test).value_counts()
        print(f"\n  Distribuicao Treino: Vitorias={train_dist.get(1, 0)}, Derrotas={train_dist.get(0, 0)}")
        print(f"  Distribuicao Teste:  Vitorias={test_dist.get(1, 0)}, Derrotas={test_dist.get(0, 0)}")

    def build_model(
        self,
        hidden_layers: List[int] = [10, 5],
        learning_rate: float = 0.01,
        epochs: int = 1000
    ):
        n_features = self.X_train_scaled.shape[1]

        # Arquitetura: [input, hidden_layers..., output]
        layer_sizes = [n_features] + hidden_layers + [1]

        self.mlp = MLP(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            epochs=epochs,
            random_state=self.random_state
        )

        print(f"\n[OK] Modelo MLP criado:")
        print(f"  - Arquitetura: {layer_sizes}")
        print(f"  - Learning Rate: {learning_rate}")
        print(f"  - Epocas: {epochs}")
        params = self.mlp.get_params()
        print(f"  - Total de parametros: {params['num_parameters']}")

    def train(self, verbose: bool = True):

        if self.mlp is None:
            raise ValueError("Construa o modelo primeiro usando build_model()")

        print("\n" + "="*70)
        print("INICIANDO TREINAMENTO")
        print("="*70 + "\n")

        start_time = time.time()

        self.mlp.fit(
            self.X_train_scaled,
            self.y_train,
            X_val=self.X_test_scaled,
            y_val=self.y_test,
            verbose=verbose
        )

        self.training_time = time.time() - start_time

        print("\n" + "="*70)
        print(f"TREINAMENTO CONCLUÍDO em {self.training_time:.2f}s")
        print("="*70 + "\n")

    def evaluate(self) -> Dict:

        if self.mlp is None:
            raise ValueError("Treine o modelo primeiro usando train()")

        # Predições
        y_pred_proba = self.mlp.predict_proba(self.X_test_scaled).flatten()
        y_pred = self.mlp.predict(self.X_test_scaled)

        # Métricas
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'training_time': self.training_time
        }

        self.predictions = {
            'y_true': self.y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        return self.metrics

    def print_evaluation_report(self):
        """Imprime relatório completo de avaliação."""
        if self.metrics is None:
            self.evaluate()

        print("\n" + "="*70)
        print("RELATORIO DE AVALIACAO - MLP")
        print("="*70 + "\n")

        print("Metricas de Performance:")
        print("-" * 70)
        print(f"  Acuracia (Accuracy):   {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)")
        print(f"  Precisao (Precision):  {self.metrics['precision']:.4f}")
        print(f"  Recall (Sensibilidade): {self.metrics['recall']:.4f}")
        print(f"  F1-Score:              {self.metrics['f1_score']:.4f}")

        print(f"\nMatriz de Confusao:")
        print("-" * 70)
        cm = self.metrics['confusion_matrix']
        print(f"                    Predito")
        print(f"                 Derrota  Vitoria")
        print(f"  Real  Derrota     {cm[0,0]:3d}      {cm[0,1]:3d}")
        print(f"        Vitoria     {cm[1,0]:3d}      {cm[1,1]:3d}")

        print(f"\nTempo de Treinamento: {self.metrics['training_time']:.2f}s")

        print("\n" + "="*70 + "\n")

    def get_training_history(self) -> pd.DataFrame:

        if self.mlp is None or not self.mlp.training_history:
            return pd.DataFrame()

        history = pd.DataFrame(self.mlp.training_history)
        history['epoch'] = range(len(history))
        return history

    def predict_new_game(self, game_features: Dict[str, float]) -> Dict:

        if self.mlp is None:
            raise ValueError("Treine o modelo primeiro")

        # Preparar features
        feature_names = self.feature_selector.selected_features
        X_new = np.array([[game_features[f] for f in feature_names]])

        # Normalizar
        X_new_scaled = self.scaler.transform(X_new)

        # Predição
        proba = self.mlp.predict_proba(X_new_scaled)[0, 0]
        prediction = int(proba >= 0.5)

        return {
            'prediction': prediction,
            'prediction_label': 'Vitória' if prediction == 1 else 'Derrota',
            'probability': proba,
            'confidence': proba if prediction == 1 else 1 - proba
        }

    def plot_training_evolution(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plota evolucao do erro durante treinamento.

        Args:
            save_path: Caminho para salvar grafico
            show: Se True, exibe o grafico
        """
        if self.mlp is None or not self.mlp.training_history:
            raise ValueError("Treine o modelo primeiro")

        history = {
            'train_loss': [h['loss'] for h in self.mlp.training_history],
            'val_loss': [h['val_loss'] for h in self.mlp.training_history] if 'val_loss' in self.mlp.training_history[0] else None,
            'train_accuracy': [h['accuracy'] for h in self.mlp.training_history] if 'accuracy' in self.mlp.training_history[0] else None,
            'val_accuracy': [h['val_accuracy'] for h in self.mlp.training_history] if 'val_accuracy' in self.mlp.training_history[0] else None
        }

        return self.visualizer.plot_training_evolution(history, save_path=save_path, show=show)

    def plot_error_matrix(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plota matriz de erros (histograma + 2 scatter plots).

        Args:
            save_path: Caminho para salvar grafico
            show: Se True, exibe o grafico
        """
        if self.predictions is None:
            self.evaluate()

        y_true = self.predictions['y_true']
        y_pred = self.predictions['y_pred_proba']

        return self.visualizer.plot_error_matrix(y_true, y_pred, save_path=save_path, show=show)

    def plot_confidence_intervals(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plota intervalos de confianca para previsoes.

        Args:
            save_path: Caminho para salvar grafico
            show: Se True, exibe o grafico
        """
        if self.predictions is None:
            self.evaluate()

        predictions = self.predictions['y_pred_proba']
        return self.visualizer.plot_confidence_intervals(predictions, save_path=save_path, show=show)

    def generate_all_plots(self, save_dir: Optional[str] = None):
        """
        Gera todos os graficos de diagnostico.

        Args:
            save_dir: Diretorio para salvar graficos (opcional)
        """
        if self.mlp is None:
            raise ValueError("Treine o modelo primeiro")

        if self.predictions is None:
            self.evaluate()

        # Preparar historico
        history = {
            'train_loss': [h['loss'] for h in self.mlp.training_history],
            'val_loss': [h.get('val_loss', None) for h in self.mlp.training_history],
            'train_accuracy': [h.get('accuracy', None) for h in self.mlp.training_history],
            'val_accuracy': [h.get('val_accuracy', None) for h in self.mlp.training_history]
        }

        y_true = self.predictions['y_true']
        y_pred = self.predictions['y_pred_proba']

        self.visualizer.plot_all_diagnostics(history, y_true, y_pred, save_dir=save_dir)
