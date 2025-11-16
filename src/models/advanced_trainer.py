"""
Advanced Neural Network Training Pipeline
==========================================
Pipeline completo para treinar modelos neurais com otimizacao de hiperparametros.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import time

from .feature_selector import FeatureSelector, prepare_features_and_target
from .optuna_optimizer import OptunaOptimizer
from .neural_networks import prepare_data_for_rnn
from .visualizations import ModelVisualizer


class AdvancedNeuralTrainer:
    """
    Pipeline completo para treinamento avancado de redes neurais.

    Inclui:
    - Selecao de features
    - Normalizacao de dados
    - Otimizacao de hiperparametros com Optuna
    - Treinamento do modelo final
    - Avaliacao completa
    """

    def __init__(
        self,
        data_path: str,
        target_column: str = 'resultado',
        test_size: float = 0.2,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Inicializa o trainer.

        Args:
            data_path: Caminho para o arquivo CSV
            target_column: Nome da coluna target
            test_size: Proporcao dos dados para teste
            val_size: Proporcao dos dados de treino para validacao
            random_state: Seed para reprodutibilidade
        """
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # Componentes
        self.df = None
        self.feature_selector = FeatureSelector(target_column)
        self.scaler = StandardScaler()
        self.optimizer = None
        self.visualizer = ModelVisualizer()

        # Dados
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Resultados
        self.optimization_results = None
        self.best_model = None
        self.final_metrics = None
        self.selected_features = None

    def load_data(self) -> pd.DataFrame:
        """Carrega os dados do CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"[OK] Dados carregados: {self.df.shape[0]} amostras, {self.df.shape[1]} colunas")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo nao encontrado: {self.data_path}")

    def select_features(
        self,
        method: str = 'correlation',
        n_features: int = 6,
        custom_features: Optional[List[str]] = None
    ) -> List[str]:
        """
        Seleciona features para o modelo.

        Args:
            method: 'correlation', 'all', 'predefined', ou 'custom'
            n_features: Numero de features (para metodo correlation)
            custom_features: Lista de features customizadas

        Returns:
            Lista de features selecionadas
        """
        if self.df is None:
            raise ValueError("Carregue os dados primeiro usando load_data()")

        if method == 'correlation':
            features = self.feature_selector.select_by_correlation(
                self.df,
                n_features=n_features
            )
        elif method == 'all':
            features = self.feature_selector.get_all_features(self.df)
        elif method == 'predefined':
            features = self.feature_selector.get_predefined_features('logistic')
        elif method == 'custom' and custom_features:
            features = custom_features
        else:
            raise ValueError(f"Metodo '{method}' invalido ou features nao fornecidas")

        self.selected_features = features
        print(f"\n[OK] {len(features)} features selecionadas: {', '.join(features)}")
        return features

    def prepare_data(self, features: List[str], normalize: bool = True):
        """
        Prepara os dados para treinamento.

        Args:
            features: Lista de features a usar
            normalize: Se True, normaliza os dados
        """
        # Preparar X e y
        X, y = prepare_features_and_target(self.df, features, self.target_column)

        # Split train+val / test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Split train / val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp
        )

        # Normalizar
        if normalize:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_val = self.scaler.transform(self.X_val)
            self.X_test = self.scaler.transform(self.X_test)

        print(f"\n[OK] Dados preparados:")
        print(f"  - Treino:    {self.X_train.shape[0]} amostras")
        print(f"  - Validacao: {self.X_val.shape[0]} amostras")
        print(f"  - Teste:     {self.X_test.shape[0]} amostras")
        print(f"  - Features:  {len(features)}")
        print(f"  - Normalizacao: {'Sim' if normalize else 'Nao'}")

        # Distribuicao de classes
        train_dist = pd.Series(self.y_train).value_counts()
        val_dist = pd.Series(self.y_val).value_counts()
        test_dist = pd.Series(self.y_test).value_counts()

        print(f"\n  Distribuicao:")
        print(f"    Treino:    Vitorias={train_dist.get(1, 0)}, Derrotas={train_dist.get(0, 0)}")
        print(f"    Validacao: Vitorias={val_dist.get(1, 0)}, Derrotas={val_dist.get(0, 0)}")
        print(f"    Teste:     Vitorias={test_dist.get(1, 0)}, Derrotas={test_dist.get(0, 0)}")

    def optimize_hyperparameters(
        self,
        model_types: List[str] = ['dense', 'lstm', 'gru'],
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Otimiza hiperparametros para diferentes tipos de modelos.

        Args:
            model_types: Lista de tipos de modelo para otimizar
            n_trials: Numero TOTAL de trials (distribuidos entre todos os modelos)
            timeout: Timeout em segundos

        Returns:
            Dicionario com resultados de otimizacao
        """
        if self.X_train is None:
            raise ValueError("Prepare os dados primeiro usando prepare_data()")

        print(f"\n{'='*70}")
        print("INICIO DA OTIMIZACAO DE HIPERPARAMETROS")
        print(f"{'='*70}")
        print(f"\nModelos a testar: {', '.join([m.upper() for m in model_types])}")
        print(f"Total de trials: {n_trials}")
        print(f"(Optuna distribuira automaticamente entre os modelos)\n")

        start_time = time.time()

        # Criar optimizer unico
        optimizer = OptunaOptimizer(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            random_state=self.random_state
        )

        # Otimizar todos os modelos em um unico estudo
        result = optimizer.optimize(
            model_types=model_types,
            n_trials=n_trials,
            timeout=timeout,
            show_progress=True
        )

        total_time = time.time() - start_time

        best_model_type = result['model_type']
        best_score = result['best_value']
        trials_per_model = result.get('trials_per_model', {})

        print(f"\n{'='*70}")
        print("RESUMO DA OTIMIZACAO")
        print(f"{'='*70}\n")
        print(f"Trials Completados por Modelo:")
        for model_type, count in sorted(trials_per_model.items()):
            print(f"  {model_type.upper():8s}: {count} trials")

        print(f"\n{'='*70}")
        print(f"MELHOR MODELO: {best_model_type.upper()}")
        print(f"Acuracia de Validacao: {best_score:.4f}")
        print(f"Tempo Total: {total_time:.2f}s")
        print(f"{'='*70}\n")

        self.optimization_results = {
            'best_model_type': best_model_type,
            'best_score': best_score,
            'total_time': total_time,
            'trials_per_model': trials_per_model,
            'n_trials_total': result['n_trials']
        }

        # Salvar optimizer
        self.optimizer = optimizer

        return self.optimization_results

    def train_best_model(self, epochs: int = 300, verbose: int = 1) -> Tuple:
        """
        Treina o modelo final com os melhores hiperparametros.

        Args:
            epochs: Numero de epocas
            verbose: Verbosidade

        Returns:
            Tupla (modelo, metricas)
        """
        if self.optimization_results is None:
            raise ValueError("Execute optimize_hyperparameters() primeiro")

        best_model_type = self.optimization_results['best_model_type']

        print(f"\n{'='*70}")
        print(f"TREINANDO MODELO FINAL: {best_model_type.upper()}")
        print(f"{'='*70}\n")

        # Treinar
        model, metrics = self.optimizer.train_best_model(
            model_type=best_model_type,
            X_test=self.X_test,
            y_test=self.y_test,
            epochs=epochs,
            verbose=verbose
        )

        self.best_model = model
        self.final_metrics = metrics
        self.final_metrics['model_type'] = best_model_type

        return model, metrics

    def predict_new_game(self, game_features: Dict[str, float]) -> Dict:
        """
        Faz predicao para um novo jogo.

        Args:
            game_features: Dicionario com valores das features

        Returns:
            Dicionario com predicao e probabilidade
        """
        if self.best_model is None:
            raise ValueError("Treine o modelo primeiro")

        # Preparar features
        X_new = np.array([[game_features[f] for f in self.selected_features]])

        # Normalizar
        X_new_scaled = self.scaler.transform(X_new)

        # Preparar para RNN se necessario
        if self.final_metrics['model_type'] in ['rnn', 'lstm', 'gru']:
            X_new_scaled = prepare_data_for_rnn(X_new_scaled)

        # Predicao
        proba = self.best_model.predict(X_new_scaled, verbose=0).flatten()[0]
        prediction = int(proba >= 0.5)

        return {
            'prediction': prediction,
            'prediction_label': 'Vitoria' if prediction == 1 else 'Derrota',
            'probability': proba,
            'confidence': proba if prediction == 1 else 1 - proba,
            'model_type': self.final_metrics['model_type']
        }

    def get_summary(self) -> Dict:
        """
        Retorna resumo completo do treinamento.

        Returns:
            Dicionario com resumo
        """
        if self.optimization_results is None or self.final_metrics is None:
            return {}

        return {
            'features': self.selected_features,
            'n_features': len(self.selected_features),
            'model_type': self.final_metrics['model_type'],
            'best_params': self.optimization_results['all_results'][
                self.final_metrics['model_type']
            ]['best_params'],
            'optimization_time': self.optimization_results['total_time'],
            'test_metrics': {
                'accuracy': self.final_metrics['accuracy'],
                'precision': self.final_metrics['precision'],
                'recall': self.final_metrics['recall'],
                'f1_score': self.final_metrics['f1_score'],
                'r2_score': self.final_metrics['r2_score']
            },
            'model_summary': self.final_metrics['model_summary']
        }

    def save_model(self, filepath: str):
        """Salva o modelo treinado."""
        if self.best_model is None:
            raise ValueError("Treine o modelo primeiro")

        self.best_model.save(filepath)
        print(f"[OK] Modelo salvo em: {filepath}")

    def load_model(self, filepath: str):
        """Carrega um modelo salvo."""
        import tensorflow as tf
        self.best_model = tf.keras.models.load_model(filepath)
        print(f"[OK] Modelo carregado de: {filepath}")

    def plot_training_evolution(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plota evolucao do erro durante treinamento.

        Args:
            save_path: Caminho para salvar grafico
            show: Se True, exibe o grafico
        """
        if self.final_metrics is None or 'history' not in self.final_metrics:
            raise ValueError("Treine o modelo primeiro usando train_best_model()")

        history = self.final_metrics['history']
        return self.visualizer.plot_training_evolution(history, save_path=save_path, show=show)

    def plot_error_matrix(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plota matriz de erros (histograma + 2 scatter plots).

        Args:
            save_path: Caminho para salvar grafico
            show: Se True, exibe o grafico
        """
        if self.best_model is None:
            raise ValueError("Treine o modelo primeiro")

        # Preparar dados para predicao
        X_test_pred = self.X_test
        if self.final_metrics['model_type'] in ['rnn', 'lstm', 'gru']:
            X_test_pred = prepare_data_for_rnn(self.X_test)

        # Fazer predicoes
        y_pred = self.best_model.predict(X_test_pred, verbose=0).flatten()

        return self.visualizer.plot_error_matrix(self.y_test, y_pred, save_path=save_path, show=show)

    def plot_confidence_intervals(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plota intervalos de confianca para previsoes.

        Args:
            save_path: Caminho para salvar grafico
            show: Se True, exibe o grafico
        """
        if self.best_model is None:
            raise ValueError("Treine o modelo primeiro")

        # Preparar dados para predicao
        X_test_pred = self.X_test
        if self.final_metrics['model_type'] in ['rnn', 'lstm', 'gru']:
            X_test_pred = prepare_data_for_rnn(self.X_test)

        # Fazer predicoes
        predictions = self.best_model.predict(X_test_pred, verbose=0).flatten()

        return self.visualizer.plot_confidence_intervals(predictions, save_path=save_path, show=show)

    def generate_all_plots(self, save_dir: Optional[str] = None):
        """
        Gera todos os graficos de diagnostico.

        Args:
            save_dir: Diretorio para salvar graficos (opcional)
        """
        if self.best_model is None or self.final_metrics is None:
            raise ValueError("Treine o modelo primeiro")

        # Preparar dados para predicao
        X_test_pred = self.X_test
        if self.final_metrics['model_type'] in ['rnn', 'lstm', 'gru']:
            X_test_pred = prepare_data_for_rnn(self.X_test)

        # Fazer predicoes
        y_pred = self.best_model.predict(X_test_pred, verbose=0).flatten()

        # Historico
        history = self.final_metrics['history']

        self.visualizer.plot_all_diagnostics(history, self.y_test, y_pred, save_dir=save_dir)
