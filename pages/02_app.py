import streamlit as st
import pandas as pd
import openai
import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega o dataframe
df = st.session_state["data"]

# Converte a coluna 'Time' para o formato de data
df["Time"] = pd.to_datetime(df["Time"])

# Campo de entrada para a filtragem no sidebar
with st.sidebar:
    fraude_mes = st.slider("Filtragem de ocorrência de fraude por mês:", 1, 12, 1)
    df_filtered = df[(df["Time"].dt.month == fraude_mes) & (df["Class"] == 1)]
    st.write(
        f"Ocorreram {len(df_filtered)} fraudes em {datetime.date(1970, fraude_mes, 1).strftime('%B')}"
    )

# Primeira linha com uma coluna
col1 = st.columns(1)[0]


# Adiciona espaçamento entre os gráficos

st.markdown("<br>", unsafe_allow_html=True)

# Geração do gráfico do slider fora do bloco condicional
fraud_months_count = df_filtered["Time"].dt.month.value_counts().sort_index()
chart_data = pd.DataFrame(
    {
        "month": range(1, 13),
        "fraude": fraud_months_count.reindex(range(1, 13), fill_value=0),
        "não fraude": df_filtered[df_filtered["Class"] == 0]["Time"]
        .dt.month.value_counts()
        .sort_index()
        .reindex(range(1, 13), fill_value=0),
    }
)
# Adiciona um título ao gráfico
col1.title("Contagem de Ocorrências de Fraude e Não Fraude ao Longo do Ano")

# Widget de gráfico dinâmico fora do bloco condicional
col1.area_chart(chart_data.set_index("month"))

# Segunda linha
col1, col2, col3 = st.columns([1, 3, 1])

# Título para o campo de entrada
col2.title("Modelo Preditivo integrado")

# Campo de entrada para valores das variáveis V1 até V28
input_values = col2.text_input(
    "Informe os dados seguindo o padrão:",
    placeholder="",
)

# Carrega o modelo treinado
modelo_treinado = joblib.load("modelo_treinado.joblib")

# Botão para executar a análise ao lado do campo de entrada
if col2.button("Executar Análise") and input_values:
    # Converta a entrada do usuário para o formato esperado (lista de float, por exemplo)
    novo_dado = [float(valor) for valor in input_values.split(",")]

    # Faça previsões usando o modelo treinado
    resultado_predicao = modelo_treinado.predict([novo_dado])

    # Organize a exibição dos resultados na barra lateral
    col2.write("\n\n\n")
    col2.write("Resultado da Análise:")
    if resultado_predicao[0] == 1:
        col2.write("Os dados são considerados uma fraude.")
    else:
        col2.write("Os dados não são considerados uma fraude.")
# IA Generativa
st.sidebar.title("💬 Chatbot")
openai_api_key = st.sidebar.text_input(
    "IA Generativa - OpenAI API Key", key="chatbot_api_key", type="password"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Como posso te ajudar?"}]

for msg in st.session_state.messages:
    st.sidebar.write(msg["content"])

if prompt := st.sidebar.text_input("Você:", key="user_input"):
    if not openai_api_key:
        st.sidebar.info("Por favor, adicione sua chave da API da OpenAI para continuar.")
        st.stop()

    openai.api_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt='\n'.join(msg["content"] for msg in st.session_state.messages),
        max_tokens=100,
    )
    msg = {"role": "assistant", "content": response.choices[0].text.strip()}
    st.session_state.messages.append(msg)
    st.sidebar.write(msg["content"])
