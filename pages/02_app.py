import streamlit as st
import pandas as pd
import openai
import datetime
import joblib


# Carrega o dataframe
df = st.session_state["data"]

# Converte a coluna 'Time' para o formato de data
df["Time"] = pd.to_datetime(df["Time"])

st.title("Utilize os filtros de análise para gerar interações gráficas:")
st.markdown(
    "Recurso interativo em uma aplicação ou plataforma que permite aos usuários analisar e visualizar a incidência de casos de fraude ao longo dos meses do ano. Por meio desse filtro, os usuários podem ajustar a exibição dos dados para focar em um mês específico, possibilitando uma análise mais detalhada da distribuição temporal das fraudes."
)

# Filtro slider
fraude_mes = st.sidebar.slider("Filtragem de ocorrência de fraude por mês:", 1, 12, 1)
df_filtered = df[(df["Time"].dt.month == fraude_mes) & (df["Class"] == 1)]
st.sidebar.write(
    "Ocorreram",
    len(df_filtered),
    "fraudes em",
    datetime.date(1970, fraude_mes, 1).strftime("%B"),
)

st.sidebar.markdown("---")

# Campo de entrada para valores das variáveis V1 até V28
input_values = st.sidebar.text_input(
    "Modelo Preditivo integrado:",
    placeholder="Informe os dados",
)

# Geração do gráfico do slider fora do bloco condicional
fraud_months_count = df_filtered["Time"].dt.month.value_counts().sort_index()
chart_data = pd.DataFrame(
    {
        "month": range(1, 13),
        "fraudulentas": fraud_months_count.reindex(range(1, 13), fill_value=0),
        "não fraudulentas": df_filtered[df_filtered["Class"] == 0]["Time"]
        .dt.month.value_counts()
        .sort_index()
        .reindex(range(1, 13), fill_value=0),
    }
)

# Widget de gráfico dinâmico fora do bloco condicional
st.area_chart(chart_data.set_index("month"))


# Carrega o modelo treinado
modelo_treinado = joblib.load("modelo_treinado.joblib")

# Botão para executar a análise
if st.sidebar.button("Executar Análise") and input_values:
    # Converta a entrada do usuário para o formato esperado (lista de float, por exemplo)
    novo_dado = [float(valor) for valor in input_values.split(",")]

    # Faça previsões usando o modelo treinado
    resultado_predicao = modelo_treinado.predict([novo_dado])

    # Organize a exibição dos resultados na barra lateral
    st.sidebar.write("\n\n\n")
    st.sidebar.write("Resultado da Análise:")
    if resultado_predicao[0] == 1:
        st.sidebar.write("Os dados são considerados uma fraude.")
    else:
        st.sidebar.write("Os dados não são considerados uma fraude.")


st.sidebar.markdown("---")


# IA Generativa

with st.sidebar:
    openai_api_key = st.text_input(
        "IA Generativa - OpenAI API Key", key="chatbot_api_key", type="password"
    )


st.title("💬 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Como posso te ajudar?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    openai.api_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=st.session_state.messages
    )
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)
