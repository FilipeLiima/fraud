import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown
import joblib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# Carregamento dos dados
# URL do arquivo no Google Drive
google_drive_url = "https://drive.google.com/uc?id=1t3JCQTfZa2qMVb0eRHPOH5d5D5XW_YIk"

# Carregue o arquivo CSV usando o Pandas
df = pd.read_csv(google_drive_url)

# Converte a coluna 'Time' para o formato de data
df["Time"] = pd.to_datetime(df["Time"])

# Ajuste outras colunas conforme necessário
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# Divisão do conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Criação do modelo preditivo
modelo_treinado = RandomForestClassifier()
modelo_treinado.fit(X_train, y_train)

# Faça previsões usando os valores de teste
y_pred = modelo_treinado.predict(X_test)

# Avalia o modelo no conjunto de teste
acuracia = modelo_treinado.score(X_test, y_test)

# Matriz de confusão
CM = confusion_matrix(y_test, y_pred)
sns.heatmap(CM, cmap="coolwarm_r", annot=True, linewidths=0.5)
plt.title("Matriz de Confusão")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.show()

# Relatório de classificação
print("\n----------Relatório de Classificação------------------------------------")
print(classification_report(y_test, y_pred))

# Exibe a acurácia
print(f"Acurácia do modelo: {acuracia * 100:.2f}%")

# Salvar o modelo treinado
joblib.dump(modelo_treinado, "modelo_treinado.joblib")
