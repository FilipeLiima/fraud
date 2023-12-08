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


# Dataset: Fraudes em Cartões de Crédito

# O conjunto de dados utilizado nesta análise refere-se a transações efetuadas por cartões de crédito em setembro de 2013 por titulares de cartões europeus.

# Overview do Dataset:

# - Total de transações: 284.807
# - Período: Dois dias
# - Número de fraudes: 492
# - Classe positiva (fraudes): 0,172% do total de transações

# Características do Dataset:

# - Variáveis numéricas de entrada
# - Resultado de uma transformação PCA (Principal Component Analysis)
# - 14 atributos principais selecionados para este projeto
# - Atributos originais não fornecidos devido a questões de confidencialidade

# Atributos Principais:

# 1. 'V1' a 'V28': Principais componentes obtidos com PCA
# 2. 'Tempo': Segundos decorridos entre cada transação e a primeira transação
# 3. 'Quantidade': Valor da transação
# 4. 'Classe': Variável de resposta - 1 em caso de fraude, 0 caso contrário

# Desequilíbrio de Classe:

# Devido ao desequilíbrio, a classe positiva (fraudes) representa 0,172% de todas as transações. Recomenda-se medir a acurácia usando a Área sob a Curva de Precisão-Recordação (AUPRC) devido ao desbalanceamento.

#Recomendação:

# Dada a natureza sensível das informações, os atributos originais foram anonimizados para garantir privacidade.

# Esse conjunto de dados é amplamente utilizado em pesquisas sobre detecção de fraudes e fornece uma base sólida para a construção de modelos preditivos eficazes.


# Carregamento dos dados
# URL do arquivo no Google Drive
google_drive_url = "https://drive.google.com/uc?id=1t3JCQTfZa2qMVb0eRHPOH5d5D5XW_YIk"

# Carregue o arquivo CSV usando o Pandas
df = pd.read_csv(google_drive_url)

# Análise Descritiva
df.head(5)
df.describe()
df.columns

# Porcentagem de fraudes e não fraudes
print('Transações não fraudulentas: ', round(df['Class'].value_counts()[0] / len(df) * 100, 2), '% dos dados')
print('Transações fraudulentas: ', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '% dos dados')


# Converte a coluna 'Time' para o formato de data
df["Time"] = pd.to_datetime(df["Time"])

# Ajuste outras colunas conforme necessário
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# Divisão do conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Criação do modelo preditivo

# Modelo Preditivo: Random Forest Classifier

# O modelo escolhido para esta análise é o Random Forest Classifier. Este é um algoritmo de aprendizado de máquina baseado em ensemble que opera construindo uma "floresta" de árvores de decisão durante o treinamento. Cada árvore na floresta contribui com uma previsão e a classe final é determinada pela votação majoritária das árvores.

# Razões para Escolha do Random Forest:

# 1. Versatilidade e Robustez: O Random Forest é conhecido por ser robusto em várias situações e geralmente não requer muita sintonia de hiperparâmetros. Ele lida bem com conjuntos de dados desequilibrados e tem boa resistência ao overfitting.

# 2. Lida com Dados Não Lineares: Diferentemente de modelos lineares, o Random Forest é capaz de capturar relações não lineares entre as variáveis, tornando-se uma escolha sólida para conjuntos de dados complexos.

# 3. Tratamento Automático de Variáveis Importantes: O modelo fornece naturalmente uma estimativa da importância de cada variável na predição, permitindo uma análise mais profunda sobre quais características têm maior impacto.

# 4. Desempenho Geral: Random Forests tendem a fornecer bom desempenho "out-of-the-box", tornando-se uma escolha eficaz para tarefas de classificação como detecção de fraudes.

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
