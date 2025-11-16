"""
Advanced Neural Network Models
===============================
Implementacao de modelos de redes neurais avancadas usando TensorFlow/Keras:
- RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.models import Sequential
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Configurar para usar CPU se GPU nao disponivel
tf.config.set_visible_devices([], 'GPU')


class NeuralNetworkBuilder:
    """
    Construtor de redes neurais para classificacao binaria.

    Suporta:
    - Dense (MLP)
    - RNN (SimpleRNN)
    - LSTM
    - GRU
    """

    def __init__(self, input_shape: Tuple[int, ...], random_state: int = 42):
        """
        Inicializa o construtor.

        Args:
            input_shape: Forma da entrada (n_features,) para Dense ou (timesteps, n_features) para RNN
            random_state: Seed para reprodutibilidade
        """
        self.input_shape = input_shape
        self.random_state = random_state

        # Configurar seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def build_dense_model(
        self,
        hidden_layers: List[int],
        activation: str = 'relu',
        dropout_rate: float = 0.3,
        l2_reg: float = 0.01
    ) -> keras.Model:
        """
        Constroi modelo Dense (MLP).

        Args:
            hidden_layers: Lista com numero de neuronios em cada camada
            activation: Funcao de ativacao ('relu', 'tanh', 'sigmoid')
            dropout_rate: Taxa de dropout (0 a 1)
            l2_reg: Regularizacao L2

        Returns:
            Modelo Keras compilado
        """
        model = Sequential(name='Dense_Model')

        # Input layer
        model.add(layers.Input(shape=self.input_shape))

        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i+1}'
            ))

            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))

        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        return model

    def build_rnn_model(
        self,
        rnn_units: List[int],
        dense_units: List[int],
        activation: str = 'tanh',
        recurrent_dropout: float = 0.2,
        dropout_rate: float = 0.3,
        l2_reg: float = 0.01
    ) -> keras.Model:
        """
        Constroi modelo RNN (SimpleRNN).

        Args:
            rnn_units: Lista com numero de unidades em cada camada RNN
            dense_units: Lista com numero de neuronios nas camadas densas finais
            activation: Funcao de ativacao
            recurrent_dropout: Dropout recorrente
            dropout_rate: Dropout regular
            l2_reg: Regularizacao L2

        Returns:
            Modelo Keras compilado
        """
        model = Sequential(name='RNN_Model')

        # Input layer
        model.add(layers.Input(shape=self.input_shape))

        # RNN layers
        for i, units in enumerate(rnn_units):
            return_sequences = i < len(rnn_units) - 1

            model.add(layers.SimpleRNN(
                units,
                activation=activation,
                return_sequences=return_sequences,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'rnn_{i+1}'
            ))

            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_rnn_{i+1}'))

        # Dense layers
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i+1}'
            ))

            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}'))

        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        return model

    def build_lstm_model(
        self,
        lstm_units: List[int],
        dense_units: List[int],
        activation: str = 'tanh',
        recurrent_dropout: float = 0.2,
        dropout_rate: float = 0.3,
        l2_reg: float = 0.01
    ) -> keras.Model:
        """
        Constroi modelo LSTM.

        Args:
            lstm_units: Lista com numero de unidades em cada camada LSTM
            dense_units: Lista com numero de neuronios nas camadas densas finais
            activation: Funcao de ativacao
            recurrent_dropout: Dropout recorrente
            dropout_rate: Dropout regular
            l2_reg: Regularizacao L2

        Returns:
            Modelo Keras compilado
        """
        model = Sequential(name='LSTM_Model')

        # Input layer
        model.add(layers.Input(shape=self.input_shape))

        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1

            model.add(layers.LSTM(
                units,
                activation=activation,
                return_sequences=return_sequences,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'lstm_{i+1}'
            ))

            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_lstm_{i+1}'))

        # Dense layers
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i+1}'
            ))

            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}'))

        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        return model

    def build_gru_model(
        self,
        gru_units: List[int],
        dense_units: List[int],
        activation: str = 'tanh',
        recurrent_dropout: float = 0.2,
        dropout_rate: float = 0.3,
        l2_reg: float = 0.01
    ) -> keras.Model:
        """
        Constroi modelo GRU.

        Args:
            gru_units: Lista com numero de unidades em cada camada GRU
            dense_units: Lista com numero de neuronios nas camadas densas finais
            activation: Funcao de ativacao
            recurrent_dropout: Dropout recorrente
            dropout_rate: Dropout regular
            l2_reg: Regularizacao L2

        Returns:
            Modelo Keras compilado
        """
        model = Sequential(name='GRU_Model')

        # Input layer
        model.add(layers.Input(shape=self.input_shape))

        # GRU layers
        for i, units in enumerate(gru_units):
            return_sequences = i < len(gru_units) - 1

            model.add(layers.GRU(
                units,
                activation=activation,
                return_sequences=return_sequences,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'gru_{i+1}'
            ))

            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_gru_{i+1}'))

        # Dense layers
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i+1}'
            ))

            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}'))

        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        return model

    def compile_model(
        self,
        model: keras.Model,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'binary_crossentropy',
        metrics: Optional[List[str]] = None
    ) -> keras.Model:
        """
        Compila o modelo.

        Args:
            model: Modelo Keras
            optimizer: Otimizador ('adam', 'rmsprop', 'sgd')
            learning_rate: Taxa de aprendizado
            loss: Funcao de loss
            metrics: Lista de metricas

        Returns:
            Modelo compilado
        """
        if metrics is None:
            metrics = ['accuracy', 'AUC', 'Precision', 'Recall']

        # Configurar otimizador
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = optimizer

        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )

        return model

    def get_callbacks(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
        reduce_lr: bool = True,
        reduce_lr_patience: int = 10,
        reduce_lr_factor: float = 0.5
    ) -> List[callbacks.Callback]:
        """
        Retorna callbacks para treinamento.

        Args:
            patience: Paciencia para early stopping
            min_delta: Mudanca minima para considerar melhoria
            restore_best_weights: Restaurar melhores pesos
            reduce_lr: Usar ReduceLROnPlateau
            reduce_lr_patience: Paciencia para reducao de LR
            reduce_lr_factor: Fator de reducao de LR

        Returns:
            Lista de callbacks
        """
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=restore_best_weights,
                verbose=0
            )
        ]

        if reduce_lr:
            callback_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=reduce_lr_factor,
                    patience=reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=0
                )
            )

        return callback_list


def prepare_data_for_rnn(X: np.ndarray, timesteps: int = 1) -> np.ndarray:
    """
    Prepara dados para modelos RNN/LSTM/GRU.

    Para modelos recorrentes, precisamos reshape para (samples, timesteps, features).
    Como estamos trabalhando com dados tabulares, usamos timesteps=1.

    Args:
        X: Array de features (n_samples, n_features)
        timesteps: Numero de timesteps (default=1 para dados tabulares)

    Returns:
        Array reshaped (n_samples, timesteps, n_features)
    """
    n_samples, n_features = X.shape
    return X.reshape(n_samples, timesteps, n_features)


def get_model_summary(model: keras.Model) -> Dict:
    """
    Retorna resumo do modelo.

    Args:
        model: Modelo Keras

    Returns:
        Dicionario com informacoes do modelo
    """
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    return {
        'name': model.name,
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'non_trainable_params': int(non_trainable_params),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
