import streamlit as st
import pandas as pd
import webbrowser

if "data" not in st.session_state:
    # URL do arquivo no Google Drive
    google_drive_url = (
        "https://drive.google.com/uc?id=1t3JCQTfZa2qMVb0eRHPOH5d5D5XW_YIk"
    )

    # Carregue o arquivo CSV usando o Pandas
    df = pd.read_csv(google_drive_url)
    st.session_state["data"] = df

st.write("# PREVENÇÃO DE FRAUDES EM CARTÕES DE CRÉDITO")
st.sidebar.markdown("Desenvolvido por Filipe Lima")
btn = st.button("Acesse o código no Github")
if btn:
    webbrowser.open_new_tab("https://github.com/FilipeLiima/fraud")

# Adiciona o texto à coluna 1
st.markdown(
    """Este projeto em Streamlit tem como objetivo criar uma interface interativa para análise e prevenção de fraudes em cartões de crédito. A aplicação oferece um painel intuitivo com gráficos interativos e estatísticas resumidas para destacar padrões e anomalias nas transações.

Utilizando algoritmos avançados de análise de padrões, o sistema identifica comportamentos usuais e transações fora do comum. Integração de modelos de machine learning treinados para analisar padrões históricos e prever transações potencialmente fraudulentas.

O sistema emite alertas em tempo real para notificar usuários e administradores sobre transações suspeitas, permitindo uma resposta imediata. Além disso, mantém um registro detalhado de todas as transações para facilitar investigações.

Configurações personalizadas permitem ajustar parâmetros de detecção de fraude conforme as necessidades específicas. A segurança é priorizada, implementando medidas robustas para proteger dados sensíveis e garantir conformidade com regulamentações de privacidade.

A geração automática de relatórios analíticos fornece insights sobre tendências de fraude e a eficácia das medidas preventivas, tornando este projeto uma abordagem proativa e inovadora para a mitigação de riscos em transações financeiras."""
)
