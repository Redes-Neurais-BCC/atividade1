"""
Optuna Hyperparameter Optimization
===================================
Otimizacao automatica de hiperparametros usando Optuna para modelos neurais.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Callable
import warnings

from .neural_networks import NeuralNetworkBuilder, prepare_data_for_rnn, get_model_summary

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """
    Otimizador de hiperparametros usando Optuna.

    Busca os melhores hiperparametros para modelos:
    - Dense (MLP)
    - RNN
    - LSTM
    - GRU
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        random_state: int = 42
    ):
        """
        Inicializa o otimizador.

        Args:
            X_train: Dados de treinamento
            y_train: Labels de treinamento
            X_val: Dados de validacao
            y_val: Labels de validacao
            random_state: Seed para reprodutibilidade
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state

        self.n_features = X_train.shape[1]
        self.best_model = None
        self.best_params = None
        self.study = None

    def objective_dense(self, trial: optuna.Trial, prefix: str = 'dense_') -> float:
        """
        Funcao objetivo para otimizacao de modelo Dense.

        Args:
            trial: Trial do Optuna
            prefix: Prefixo para nomes de parametros

        Returns:
            Metrica de validacao (accuracy)
        """
        # Hiperparametros a otimizar
        n_layers = trial.suggest_int(f'{prefix}n_layers', 1, 3)
        hidden_units = [
            trial.suggest_int(f'{prefix}units_l{i}', 16, 256, log=True)
            for i in range(n_layers)
        ]

        activation = trial.suggest_categorical(f'{prefix}activation', ['relu', 'tanh', 'elu'])
        dropout_rate = trial.suggest_float(f'{prefix}dropout_rate', 0.0, 0.5)
        l2_reg = trial.suggest_float(f'{prefix}l2_reg', 1e-5, 1e-2, log=True)

        learning_rate = trial.suggest_float(f'{prefix}learning_rate', 1e-4, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical(f'{prefix}optimizer', ['adam', 'rmsprop'])
        batch_size = trial.suggest_categorical(f'{prefix}batch_size', [16, 32, 64])

        # Construir modelo
        builder = NeuralNetworkBuilder(
            input_shape=(self.n_features,),
            random_state=self.random_state
        )

        model = builder.build_dense_model(
            hidden_layers=hidden_units,
            activation=activation,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )

        model = builder.compile_model(
            model,
            optimizer=optimizer_name,
            learning_rate=learning_rate
        )

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        # Treinar
        try:
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=200,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )

            # Avaliar
            y_pred = (model.predict(self.X_val, verbose=0) > 0.5).astype(int).flatten()
            accuracy = accuracy_score(self.y_val, y_pred)

            return accuracy

        except Exception as e:
            print(f"Erro no trial: {e}")
            return 0.0

    def objective_rnn(self, trial: optuna.Trial, model_type: str = 'lstm', prefix: str = None) -> float:
        """
        Funcao objetivo para otimizacao de modelos RNN/LSTM/GRU.

        Args:
            trial: Trial do Optuna
            model_type: Tipo de modelo ('rnn', 'lstm', 'gru')
            prefix: Prefixo para nomes de parametros

        Returns:
            Metrica de validacao (accuracy)
        """
        if prefix is None:
            prefix = f'{model_type}_'

        # Preparar dados para RNN
        X_train_rnn = prepare_data_for_rnn(self.X_train)
        X_val_rnn = prepare_data_for_rnn(self.X_val)

        # Hiperparametros a otimizar
        n_rnn_layers = trial.suggest_int(f'{prefix}n_rnn_layers', 1, 2)
        rnn_units = [
            trial.suggest_int(f'{prefix}rnn_units_l{i}', 32, 128, log=True)
            for i in range(n_rnn_layers)
        ]

        n_dense_layers = trial.suggest_int(f'{prefix}n_dense_layers', 0, 2)
        dense_units = [
            trial.suggest_int(f'{prefix}dense_units_l{i}', 16, 128, log=True)
            for i in range(n_dense_layers)
        ] if n_dense_layers > 0 else []

        activation = trial.suggest_categorical(f'{prefix}activation', ['tanh', 'relu'])
        recurrent_dropout = trial.suggest_float(f'{prefix}recurrent_dropout', 0.0, 0.3)
        dropout_rate = trial.suggest_float(f'{prefix}dropout_rate', 0.0, 0.5)
        l2_reg = trial.suggest_float(f'{prefix}l2_reg', 1e-5, 1e-2, log=True)

        learning_rate = trial.suggest_float(f'{prefix}learning_rate', 1e-4, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical(f'{prefix}optimizer', ['adam', 'rmsprop'])
        batch_size = trial.suggest_categorical(f'{prefix}batch_size', [16, 32, 64])

        # Construir modelo
        builder = NeuralNetworkBuilder(
            input_shape=(1, self.n_features),
            random_state=self.random_state
        )

        if model_type == 'lstm':
            model = builder.build_lstm_model(
                lstm_units=rnn_units,
                dense_units=dense_units,
                activation=activation,
                recurrent_dropout=recurrent_dropout,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg
            )
        elif model_type == 'gru':
            model = builder.build_gru_model(
                gru_units=rnn_units,
                dense_units=dense_units,
                activation=activation,
                recurrent_dropout=recurrent_dropout,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg
            )
        else:  # rnn
            model = builder.build_rnn_model(
                rnn_units=rnn_units,
                dense_units=dense_units,
                activation=activation,
                recurrent_dropout=recurrent_dropout,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg
            )

        model = builder.compile_model(
            model,
            optimizer=optimizer_name,
            learning_rate=learning_rate
        )

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        # Treinar
        try:
            history = model.fit(
                X_train_rnn, self.y_train,
                validation_data=(X_val_rnn, self.y_val),
                epochs=200,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )

            # Avaliar
            y_pred = (model.predict(X_val_rnn, verbose=0) > 0.5).astype(int).flatten()
            accuracy = accuracy_score(self.y_val, y_pred)

            return accuracy

        except Exception as e:
            print(f"Erro no trial: {e}")
            return 0.0

    def objective_multi_model(self, trial: optuna.Trial, model_types: List[str]) -> float:
        """
        Funcao objetivo que testa multiplos tipos de modelo.

        Args:
            trial: Trial do Optuna
            model_types: Lista de tipos de modelo a testar

        Returns:
            Metrica de validacao (accuracy)
        """
        # Escolher tipo de modelo para este trial
        model_type = trial.suggest_categorical('model_type', model_types)

        # Executar objetivo especifico do modelo com prefixo unico
        prefix = f'{model_type}_'

        if model_type == 'dense':
            return self.objective_dense(trial, prefix=prefix)
        else:
            return self.objective_rnn(trial, model_type=model_type, prefix=prefix)

    def optimize(
        self,
        model_types: List[str] = ['dense', 'lstm', 'gru'],
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress: bool = True
    ) -> Dict:
        """
        Executa a otimizacao de hiperparametros para multiplos modelos.

        Args:
            model_types: Lista de tipos de modelo ('dense', 'rnn', 'lstm', 'gru')
            n_trials: Numero TOTAL de trials (distribuidos entre todos os modelos)
            timeout: Timeout em segundos
            n_jobs: Numero de jobs paralelos
            show_progress: Mostrar barra de progresso

        Returns:
            Dicionario com melhores parametros e resultados
        """
        print(f"\n{'='*70}")
        print(f"INICIANDO OTIMIZACAO - Modelos: {', '.join([m.upper() for m in model_types])}")
        print(f"{'='*70}\n")
        print(f"Total de trials: {n_trials}")
        print(f"Distribuidos entre {len(model_types)} modelos\n")

        # Configurar sampler e pruner
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

        # Criar study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )

        # Definir funcao objetivo
        objective_func = lambda trial: self.objective_multi_model(trial, model_types)

        # Otimizar
        self.study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress
        )

        # Melhores parametros
        self.best_params = self.study.best_params
        best_value = self.study.best_value
        best_model_type = self.best_params.get('model_type', model_types[0])

        print(f"\n{'='*70}")
        print(f"OTIMIZACAO CONCLUIDA")
        print(f"{'='*70}\n")
        print(f"Melhor Modelo: {best_model_type.upper()}")
        print(f"Melhor Acuracia de Validacao: {best_value:.4f}")
        print(f"\nMelhores Hiperparametros:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        # Analisar distribuicao de trials por modelo
        trials_per_model = {}
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                model = trial.params.get('model_type', 'unknown')
                trials_per_model[model] = trials_per_model.get(model, 0) + 1

        print(f"\nDistribuicao de Trials:")
        for model, count in sorted(trials_per_model.items()):
            print(f"  {model.upper()}: {count} trials")

        return {
            'best_params': self.best_params,
            'best_value': best_value,
            'n_trials': len(self.study.trials),
            'model_type': best_model_type,
            'trials_per_model': trials_per_model
        }

    def train_best_model(
        self,
        model_type: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 300,
        verbose: int = 1
    ) -> Tuple[keras.Model, Dict]:
        """
        Treina o modelo com os melhores hiperparametros encontrados.

        Args:
            model_type: Tipo de modelo
            X_test: Dados de teste
            y_test: Labels de teste
            epochs: Numero de epocas
            verbose: Verbosidade

        Returns:
            Tupla (modelo treinado, metricas)
        """
        if self.best_params is None:
            raise ValueError("Execute optimize() primeiro")

        print(f"\n{'='*70}")
        print(f"TREINANDO MODELO FINAL COM MELHORES HIPERPARAMETROS")
        print(f"{'='*70}\n")

        # Extrair parametros com prefixo do tipo de modelo
        all_params = self.best_params
        prefix = f'{model_type}_'

        # Filtrar parametros do modelo especifico
        params = {}
        for key, value in all_params.items():
            if key.startswith(prefix):
                # Remover prefixo
                clean_key = key[len(prefix):]
                params[clean_key] = value
            elif key == 'model_type':
                # Ignorar model_type
                continue
            else:
                # Manter parametros sem prefixo
                params[key] = value

        # Construir modelo baseado no tipo
        if model_type == 'dense':
            builder = NeuralNetworkBuilder(
                input_shape=(self.n_features,),
                random_state=self.random_state
            )

            # Extrair arquitetura
            n_layers = params['n_layers']
            hidden_units = [params[f'units_l{i}'] for i in range(n_layers)]

            model = builder.build_dense_model(
                hidden_layers=hidden_units,
                activation=params['activation'],
                dropout_rate=params['dropout_rate'],
                l2_reg=params['l2_reg']
            )

            X_train_final = self.X_train
            X_test_final = X_test

        else:  # RNN/LSTM/GRU
            # Preparar dados
            X_train_final = prepare_data_for_rnn(self.X_train)
            X_test_final = prepare_data_for_rnn(X_test)

            builder = NeuralNetworkBuilder(
                input_shape=(1, self.n_features),
                random_state=self.random_state
            )

            # Extrair arquitetura
            n_rnn_layers = params['n_rnn_layers']
            rnn_units = [params[f'rnn_units_l{i}'] for i in range(n_rnn_layers)]

            n_dense_layers = params.get('n_dense_layers', 0)
            dense_units = [
                params[f'dense_units_l{i}'] for i in range(n_dense_layers)
            ] if n_dense_layers > 0 else []

            if model_type == 'lstm':
                model = builder.build_lstm_model(
                    lstm_units=rnn_units,
                    dense_units=dense_units,
                    activation=params['activation'],
                    recurrent_dropout=params['recurrent_dropout'],
                    dropout_rate=params['dropout_rate'],
                    l2_reg=params['l2_reg']
                )
            elif model_type == 'gru':
                model = builder.build_gru_model(
                    gru_units=rnn_units,
                    dense_units=dense_units,
                    activation=params['activation'],
                    recurrent_dropout=params['recurrent_dropout'],
                    dropout_rate=params['dropout_rate'],
                    l2_reg=params['l2_reg']
                )
            else:
                model = builder.build_rnn_model(
                    rnn_units=rnn_units,
                    dense_units=dense_units,
                    activation=params['activation'],
                    recurrent_dropout=params['recurrent_dropout'],
                    dropout_rate=params['dropout_rate'],
                    l2_reg=params['l2_reg']
                )

        # Compilar
        model = builder.compile_model(
            model,
            optimizer=params['optimizer'],
            learning_rate=params['learning_rate']
        )

        # Callbacks
        callback_list = builder.get_callbacks(
            patience=20,
            reduce_lr=True,
            reduce_lr_patience=10
        )

        # Treinar
        history = model.fit(
            X_train_final, self.y_train,
            validation_split=0.15,
            epochs=epochs,
            batch_size=params['batch_size'],
            callbacks=callback_list,
            verbose=verbose
        )

        # Avaliar no conjunto de teste
        y_pred_proba = model.predict(X_test_final, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'r2_score': r2_score(y_test, y_pred_proba),
            'model_summary': get_model_summary(model),
            'history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            }
        }

        self.best_model = model

        print(f"\n{'='*70}")
        print("METRICAS NO CONJUNTO DE TESTE")
        print(f"{'='*70}\n")
        print(f"  Acuracia:  {metrics['accuracy']:.4f}")
        print(f"  Precisao:  {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  R2-Score:  {metrics['r2_score']:.4f}")
        print(f"\n{'='*70}\n")

        return model, metrics

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Retorna historico de otimizacao como DataFrame.

        Returns:
            DataFrame com trials
        """
        if self.study is None:
            return pd.DataFrame()

        df = self.study.trials_dataframe()
        return df

    def plot_optimization_history(self):
        """Plota historico de otimizacao (requires plotly)."""
        if self.study is None:
            raise ValueError("Execute optimize() primeiro")

        from optuna.visualization import plot_optimization_history, plot_param_importances

        fig1 = plot_optimization_history(self.study)
        fig2 = plot_param_importances(self.study)

        return fig1, fig2
