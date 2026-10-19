import streamlit as st
import pandas as pd
import requests
import io
import base64

# 1. Configurazione della pagina
st.set_page_config(
    page_title="Tabular VAE & LLM Analyst", 
    page_icon="🤖", 
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8000"

st.title("🤖 Tabular VAE & Natural Language Analyst")
st.markdown("Piattaforma per la generazione di dati sintetici e analisi guidata da AI.")

# Inizializzazione dello stato della sessione (session_state)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "real_df" not in st.session_state:
    st.session_state.real_df = None
if "synthetic_df" not in st.session_state:
    st.session_state.synthetic_df = None

# ---------------------------------------------------------
# SIDEBAR: Workflow Upload, Training e Generazione
# ---------------------------------------------------------
st.sidebar.header("📁 1. Upload Dataset Reale")

uploaded_file = st.sidebar.file_uploader(
    "Carica il tuo file CSV (Drag & Drop)", 
    type=["csv"]
)

if uploaded_file is not None:
    if st.sidebar.button("Invia File al Backend", use_container_width=True):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        try:
            res = requests.post(f"{BACKEND_URL}/upload", files=files)
            if res.status_code == 200:
                st.sidebar.success("File caricato con successo!")
                # Salva il dataframe nello stato locale per la visualizzazione immediata
                uploaded_file.seek(0)
                st.session_state.real_df = pd.read_csv(uploaded_file)
            else:
                st.sidebar.error(f"Errore Upload: {res.json().get('detail')}")
        except Exception as e:
            st.sidebar.error(f"Impossibile connettersi al backend: {e}")

st.sidebar.divider()
st.sidebar.header("⚙️ 2. Addestramento VAE")

epochs = st.sidebar.number_input("Numero di Epoche", min_value=10, max_value=500, value=50, step=10)
latent_dim = st.sidebar.number_input("Dimensione Spazio Latente", min_value=2, max_value=32, value=4, step=1)
batch_size = st.sidebar.number_input("Batch Size", min_value=16, max_value=256, value=64, step=16)

if st.sidebar.button("Avvia Addestramento VAE", use_container_width=True):
    with st.spinner("Addestramento del VAE in corso..."):
        try:
            res = requests.post(
                f"{BACKEND_URL}/train", 
                json={"epochs": epochs, "latent_dim": latent_dim, "batch_size": batch_size}
            )
            if res.status_code == 200:
                st.sidebar.success("Model Training Completato!")
                st.sidebar.json(res.json().get("metrics"))
            else:
                st.sidebar.error(f"Errore Training: {res.json().get('detail')}")
        except Exception as e:
            st.sidebar.error(f"Errore di connessione: {e}")

st.sidebar.divider()
st.sidebar.header("🎲 3. Generazione Dati Sintetici")

num_samples = st.sidebar.number_input("Numero di campioni da generare", min_value=10, max_value=10000, value=100, step=50)

if st.sidebar.button("Genera Dataset Sintetico", use_container_width=True):
    with st.spinner("Generazione dei campioni dallo spazio latente..."):
        try:
            res = requests.post(f"{BACKEND_URL}/generate", json={"num_samples": num_samples})
            if res.status_code == 200:
                # Carica il CSV ricevuto in memoria
                csv_bytes = res.content
                st.session_state.synthetic_df = pd.read_csv(io.BytesIO(csv_bytes))
                st.sidebar.success("Dati sintetici generati!")
                
                # Bottone per il download del file CSV sintetico
                st.sidebar.download_button(
                    label="📥 Scarica CSV Sintetico",
                    data=csv_bytes,
                    file_name="synthetic_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.sidebar.error(f"Errore Generazione: {res.json().get('detail')}")
        except Exception as e:
            st.sidebar.error(f"Errore di connessione: {e}")

# ---------------------------------------------------------
# MAIN AREA: Tabella Comparativa e Chatbot
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Tabella Comparativa (Reali vs Sintetici)", "💬 Chatbot LLM Analyst"])

# TAB 1: Tabella Comparativa Dati Reali vs Sintetici
with tab1:
    st.header("Confronto Dati Reali vs Sintetici")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dati Reali")
        if st.session_state.real_df is not None:
            st.write("--- Anteprima dei Dati ---")
            st.dataframe(st.session_state.real_df.head(10), use_container_width=True)
            st.write("--- Statistiche Descrittive ---")
            st.dataframe(st.session_state.real_df.describe(), use_container_width=True)
        else:
            st.info("Carica un file CSV dalla sidebar per visualizzare i dati reali.")
            
    with col2:
        st.subheader("Dati Sintetici (Generati)")
        if st.session_state.synthetic_df is not None:
            st.write("--- Anteprima dei Dati ---")
            st.dataframe(st.session_state.synthetic_df.head(10), use_container_width=True)
            st.write("--- Statistiche Descrittive ---")
            st.dataframe(st.session_state.synthetic_df.describe(), use_container_width=True)
        else:
            st.info("Addestra il modello e genera i dati sintetici dalla sidebar per abilitare il confronto.")

# TAB 2: Chatbot Integrato
with tab2:
    st.header("Chatbot Natural Language Analyst")
    st.caption("Fai domande sui dati sintetici (es. 'Qual è la correlazione tra reddito e risparmio?')")
    
    # Rende visibili le precedenti interazioni
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "code" in message and message["code"]:
                with st.expander("Mostra Codice Python Generato"):
                    st.code(message["code"], language="python")
            if "image" in message and message["image"]:
                st.image(base64.b64decode(message["image"]))

    # Input della Chat
    if prompt := st.chat_input("Scrivi la tua richiesta di analisi..."):
        # Aggiunge la domanda dell'utente allo stato
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Risposta dell'assistente
        with st.chat_message("assistant"):
            with st.spinner("L'LLM sta analizzando la richiesta ed eseguendo il codice..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/api/analyze", json={"prompt": prompt})
                    if res.status_code == 200:
                        data = res.json()
                        code = data.get("generated_code")
                        img_b64 = data.get("image_base64")
                        
                        st.markdown("Ecco l'analisi ottenuta:")
                        if code:
                            with st.expander("Mostra Codice Python Generato"):
                                st.code(code, language="python")
                        if img_b64:
                            st.image(base64.b64decode(img_b64))
                            
                        # Salva la risposta dell'assistente nella cronologia
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Analisi completata con successo.",
                            "code": code,
                            "image": img_b64
                        })
                    else:
                        err_msg = f"Errore Backend: {res.json().get('detail')}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                except Exception as e:
                    err_msg = f"Impossibile contattare il server Backend: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})