import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class FeatureSelector:
    def __init__(self, target_column: str = 'resultado'):
        self.target_column = target_column
        self.correlation_scores = None
        self.selected_features = None

    def calculate_correlation(self, df: pd.DataFrame) -> pd.Series:
        if self.target_column not in df.columns:
            raise ValueError(f"Coluna target '{self.target_column}' não encontrada no DataFrame")

        # Calcular matriz de correlação
        correlation_matrix = df.corr(numeric_only=True)

        # Extrair correlações com o target
        target_correlation = correlation_matrix[self.target_column].drop(
            self.target_column, errors='ignore'
        )

        # Ordenar por valor absoluto (correlação mais forte)
        self.correlation_scores = target_correlation.abs().sort_values(ascending=False)

        return target_correlation.sort_values(ascending=False)

    def select_by_correlation(
        self,
        df: pd.DataFrame,
        n_features: int = 6,
        exclude_columns: Optional[List[str]] = None
    ) -> List[str]:
        if exclude_columns is None:
            exclude_columns = ['data-jogo']  # Excluir data por padrão

        # Calcular correlação se ainda não foi calculada
        if self.correlation_scores is None:
            self.calculate_correlation(df)

        # Filtrar colunas excluídas
        available_features = self.correlation_scores.drop(
            exclude_columns, errors='ignore'
        )

        # Selecionar top N
        self.selected_features = available_features.head(n_features).index.tolist()

        return self.selected_features

    def get_all_features(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> List[str]:
        if exclude_columns is None:
            exclude_columns = ['data-jogo']

        # Todas as colunas numéricas exceto target e excluídas
        all_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remover target e colunas excluídas
        features = [
            col for col in all_columns
            if col != self.target_column and col not in exclude_columns
        ]

        self.selected_features = features
        return features

    def get_predefined_features(self, feature_set: str = 'logistic') -> List[str]:

        predefined_sets = {
            'logistic': [
                'porcentagem-triplos',
                'triplos-convertidos',
                'rebotes-totais',
                'porcentagem-arremessos',
                'assistencias',
                'mando-de-jogo'
            ],
            'linear': [
                'arremessos-convertidos',
                'porcentagem-arremessos',
                'triplos-convertidos',
                'assistencias'
            ],
            'combined': [
                'porcentagem-triplos',
                'triplos-convertidos',
                'rebotes-totais',
                'porcentagem-arremessos',
                'assistencias',
                'mando-de-jogo',
                'arremessos-convertidos'
            ]
        }

        if feature_set not in predefined_sets:
            raise ValueError(f"Feature set '{feature_set}' não encontrado. "
                           f"Opções: {list(predefined_sets.keys())}")

        self.selected_features = predefined_sets[feature_set]
        return self.selected_features

    def print_correlation_report(self, df: pd.DataFrame, top_n: int = 10):
        correlation = self.calculate_correlation(df)

        print(f"\n{'='*70}")
        print(f"RELATÓRIO DE CORRELAÇÃO COM '{self.target_column.upper()}'")
        print(f"{'='*70}\n")

        print(f"Top {top_n} Features (Correlação Positiva):")
        print("-" * 70)
        positive_corr = correlation[correlation > 0].head(top_n)
        for i, (feature, corr) in enumerate(positive_corr.items(), 1):
            print(f"{i:2d}. {feature:30s} | Correlação: {corr:+.4f}")

        print(f"\nTop {top_n} Features (Correlação Negativa):")
        print("-" * 70)
        negative_corr = correlation[correlation < 0].tail(top_n)
        for i, (feature, corr) in enumerate(negative_corr.items(), 1):
            print(f"{i:2d}. {feature:30s} | Correlação: {corr:+.4f}")

        print(f"\n{'='*70}\n")

    def get_feature_stats(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        stats = df[features].describe().T
        stats['missing'] = df[features].isnull().sum()

        if self.correlation_scores is not None:
            stats['correlation'] = [
                self.correlation_scores.get(f, 0) for f in features
            ]

        return stats


def prepare_features_and_target(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'resultado'
) -> Tuple[np.ndarray, np.ndarray]:
    # Verificar se todas as features existem
    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Features não encontradas: {missing_features}")

    if target not in df.columns:
        raise ValueError(f"Target '{target}' não encontrado no DataFrame")

    X = df[features].values
    y = df[target].values

    return X, y
