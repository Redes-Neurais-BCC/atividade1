import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

TARGET_LOG = 'resultado'

FILE_PATH = r'C:\Users\pipoc\Documents\Data_Felipe\UFAPE\RNN - FM\atividade1-parte1\data\processed\dallas_games_2024-25.csv'

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo CSV não encontrado em {FILE_PATH}")
    exit()

if TARGET_LOG not in df.columns:
    print(f"\n❌ ERRO CRÍTICO: A coluna '{TARGET_LOG}' não foi encontrada no CSV.")
    exit()

correlation_matrix_log = df.corr(numeric_only=True)
target_correlation_log = correlation_matrix_log[TARGET_LOG].sort_values(ascending=False)

print(f"\n--- Coeficientes de Correlação com a Variável TARGET ({TARGET_LOG}) ---")
print(target_correlation_log.drop(TARGET_LOG, errors='ignore'))


FEATURES_LOG = [
    'porcentagem-triplos',
    'triplos-convertidos',
    'rebotes-totais',
    'porcentagem-arremessos',
    'assistencias',
    'mando-de-jogo'
]

X = df[FEATURES_LOG]
y = df[TARGET_LOG]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_log = LogisticRegression(solver='liblinear', random_state=42)
model_log.fit(X_train, y_train)

y_pred = model_log.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
intercepto = model_log.intercept_[0]

print(f"\n--- Resultados do Modelo de Regressão Logística ---")
print(f"Acurácia (Teste): {accuracy:.4f}")
print(f"Intercepto (Beta_0): {intercepto:.4f}")

coef_log = pd.Series(model_log.coef_[0], index=FEATURES_LOG)
print("\nCoeficientes (Beta_n):")
print(coef_log.sort_values(ascending=False))