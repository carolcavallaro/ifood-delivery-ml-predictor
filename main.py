"""
Projeto: Modelo de Machine Learning para Previsão de Valor de Pedidos
Autor: Carolina Cavallaro
Descrição: Algoritmo de Regressão em Python para prever o valor total de pedidos 
           com base no tempo de preparo, distância de entrega e dia da semana.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Simulação / Carga do Dataset (Dados Sintéticos de Delivery)
print("--- 1. Carregando e Processando os Dados ---")
data = {
    'distancia_km': [1.2, 3.5, 5.0, 2.1, 8.4, 6.2, 0.8, 4.3, 7.1, 2.9, 5.5, 1.8],
    'tempo_preparo_min': [15, 25, 30, 20, 45, 35, 10, 28, 40, 18, 32, 12],
    'itens_no_pedido': [2, 4, 5, 2, 7, 6, 1, 3, 6, 3, 4, 1],
    'dia_fim_de_semana': [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],  # 0 = Dia útil, 1 = Fim de semana
    'valor_total_reais': [35.0, 78.5, 95.0, 42.0, 150.0, 110.0, 22.0, 65.0, 135.0, 50.0, 88.0, 28.0]
}

df = pd.DataFrame(data)
print(df.head())

# 2. Divisão de Variáveis (Features X e Target y)
X = df[['distancia_km', 'tempo_preparo_min', 'itens_no_pedido', 'dia_fim_de_semana']]
y = df['valor_total_reais']

# 3. Separação de Dados de Treino e Teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_size=0.2, random_state=42) if 'test_test_size' in locals() else train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinamento do Modelo de Machine Learning (Random Forest Regressor)
print("--- 2. Treinando o Modelo Random Forest ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Avaliação do Modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- 3. Métrica de Desempenho ---")
print(f"Erro Médio Absoluto (MAE): R$ {mae:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

# 6. Exemplo Prático de Previsão
novo_pedido = pd.DataFrame({
    'distancia_km': [3.0],
    'tempo_preparo_min': [22],
    'itens_no_pedido': [3],
    'dia_fim_de_semana': [1]
})

previsao = model.predict(novo_pedido)
print(f"--- 4. Previsão para Novo Pedido ---")
print(f"Valor estimado do pedido: R$ {previsao[0]:.2f}")