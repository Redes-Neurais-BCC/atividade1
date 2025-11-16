import numpy as np
from typing import List, Tuple, Optional


class MLP:

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        epochs: int = 1000,
        random_state: Optional[int] = None
    ):

        if random_state is not None:
            np.random.seed(random_state)

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_layers = len(layer_sizes)

        # Inicializar pesos e vieses com valores aleatórios pequenos
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

        # Histórico de treinamento
        self.training_history = {
            'loss': [],
            'accuracy': []
        }

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Função de ativação sigmoid."""
        # Clip para evitar overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        return z * (1 - z)

    def forward_propagation(self, X: np.ndarray) -> List[np.ndarray]:

        activations = [X]

        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)

        return activations

    def backward_propagation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        m = X.shape[0]  # número de amostras

        # Inicializar gradientes
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]

        # Calcular erro da camada de saída
        delta = activations[-1] - y.reshape(-1, 1)

        # Backpropagation através das camadas
        for i in range(self.num_layers - 2, -1, -1):
            # Gradientes
            weight_gradients[i] = np.dot(activations[i].T, delta) / m
            bias_gradients[i] = np.sum(delta, axis=0, keepdims=True) / m

            # Se não for a primeira camada, propagar o erro
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])

        return weight_gradients, bias_gradients

    def update_parameters(
        self,
        weight_gradients: List[np.ndarray],
        bias_gradients: List[np.ndarray]
    ):

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        m = y_true.shape[0]
        # Adicionar epsilon para evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):

        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(self.epochs):
            # Forward propagation
            activations = self.forward_propagation(X)
            y_pred = activations[-1]

            # Calcular loss e accuracy
            loss = self.compute_loss(y, y_pred)
            accuracy = self.score(X, y)

            self.training_history['loss'].append(loss)
            self.training_history['accuracy'].append(accuracy)

            # Backward propagation
            weight_gradients, bias_gradients = self.backward_propagation(X, y, activations)

            # Atualizar parâmetros
            self.update_parameters(weight_gradients, bias_gradients)

            # Validação (se fornecida)
            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(y_val, self.predict_proba(X_val))
                val_accuracy = self.score(X_val, y_val)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)

            # Imprimir progresso
            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                msg = f"Epoch {epoch}/{self.epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}"
                if X_val is not None:
                    msg += f" - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}"
                print(msg)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations = self.forward_propagation(X)
        return activations[-1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int).flatten()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_params(self) -> dict:
        return {
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'num_parameters': sum(w.size for w in self.weights) + sum(b.size for b in self.biases)
        }
