import streamlit as st
from openai import OpenAI

# --- CONFIGURAZIONE DEL CLIENT E DELL'INTERFACCIA ---

# LM Studio fornisce un endpoint locale che imita le API di OpenAI.
# Il client viene inizializzato per puntare all'indirizzo locale (localhost) sulla porta 1234.
# 'api_key' è necessaria per la struttura della libreria, ma LM Studio non la richiede realmente.
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Impostazione della configurazione della pagina Streamlit
st.set_page_config(page_title="Educational Chat Streamer")

# --- BARRA LATERALE: CONTROLLI DEL MODELLO ---

# La sidebar viene utilizzata per separare i parametri di configurazione dalla chat principale.
with st.sidebar:
    st.title("Parametri Modello")
    st.info("Regola i parametri per modificare il comportamento del linguaggio naturale.")
    
    # Lo slider imposta la 'Temperature', che determina quanto la risposta debba essere creativa o deterministica.
    # Valori bassi (0.2) portano a risposte precise; valori alti (1.2) a risposte più varie.
    temp = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
    
    # Pulsante per resettare la conversazione.
    # In Streamlit, premere un pulsante attiva il blocco di codice sottostante.
    if st.button("Cancella Cronologia", use_container_width=True):
        # Svuotiamo la lista dei messaggi salvata nello stato della sessione.
        st.session_state.messages = []
        # Forza il ricaricamento dell'app per mostrare l'interfaccia pulita.
        st.rerun()

# Titolo principale dell'applicazione.
st.title("Chat Streaming con LM Studio")
st.markdown("Applicazione per interagire con un modello linguistico locale tramite streaming dei dati.")

# --- GESTIONE DELLO STATO (SESSION STATE) ---

# Streamlit riesegue l'intero script Python ogni volta che l'utente interagisce con un elemento.
# Per non perdere la cronologia della conversazione tra un'esecuzione e l'altra,
# utilizziamo 'st.session_state', un dizionario persistente durante la sessione dell'utente.
if "messages" not in st.session_state:
    # Se la chiave 'messages' non esiste, la inizializziamo come lista vuota.
    st.session_state.messages = []

# --- VISUALIZZAZIONE DELLA CONVERSAZIONE ---

# Iteriamo sulla lista dei messaggi salvati per renderizzarli nell'interfaccia.
# Ogni messaggio è un dizionario con chiavi 'role' (user/assistant) e 'content'.
for message in st.session_state.messages:
    # 'st.chat_message' crea un contenitore grafico adatto al ruolo specificato.
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- GESTIONE DELL'INPUT E GENERAZIONE RISPOSTA ---

# 'st.chat_input' cattura il testo scritto dall'utente. Se l'utente preme invio, il prompt viene popolato.
if prompt := st.chat_input("Scrivi un messaggio..."):
    
    # 1. Registrazione del messaggio utente.
    # Aggiungiamo il prompt alla cronologia nello stato della sessione.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Visualizziamo immediatamente il messaggio appena scritto.
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generazione della risposta dell'assistente.
    with st.chat_message("assistant"):
        # Creiamo un 'placeholder' (contenitore vuoto) che aggiorneremo dinamicamente.
        # Questo permette l'effetto di streaming, dove il testo appare parola per parola.
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Chiamata al client per generare la risposta.
            # Impostando 'stream=True', il client restituisce un oggetto iterabile (generatore).
            stream = client.chat.completions.create(
                model="gpt-oss-20b", # Identificativo del modello caricato in LM Studio.
                messages=[
                    # Passiamo l'intera cronologia per dare contesto al modello.
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                temperature=temp,
                stream=True,
            )
            
            # Ciclo di lettura dei 'chunk' (frammenti di testo) inviati dal modello.
            for chunk in stream:
                # Verifichiamo la presenza di contenuto nel frammento attuale.
                if chunk.choices[0].delta.content:
                    # Accumuliamo il testo ricevuto nella stringa totale.
                    full_response += chunk.choices[0].delta.content
                    # Aggiorniamo il visualizzatore in tempo reale aggiungendo un cursore visivo.
                    response_placeholder.markdown(full_response + "|")
            
            # Una volta terminato lo streaming, mostriamo la risposta definitiva senza il cursore.
            response_placeholder.markdown(full_response)
            
            # 3. Salvataggio della risposta completa.
            # È fondamentale salvare la risposta intera in 'messages' per mantenerla nelle esecuzioni future.
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            # Gestione degli errori in caso di problemi di comunicazione con il server locale.
            st.error(f"Errore di connessione: {e}")
            st.warning("Assicurarsi che LM Studio sia attivo e che il Local Server sia in ascolto sulla porta 1234.")