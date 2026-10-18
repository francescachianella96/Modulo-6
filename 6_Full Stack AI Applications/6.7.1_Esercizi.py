import streamlit as st
import pandas as pd

# --- CONFIGURAZIONE DELLA PAGINA ---
st.set_page_config(page_title="Dashboard Analisi Sentiment")

# --- INIZIALIZZAZIONE DELLO STATO (SESSION STATE) ---
# Utilizziamo lo stato della sessione per mantenere la persistenza dei dati al refresh della pagina.
# Inizializziamo il contatore delle analisi e i risultati se non sono già presenti.
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

if "sentiment_storage" not in st.session_state:
    # Utilizziamo un dizionario per memorizzare il conteggio delle categorie
    st.session_state.sentiment_storage = {"Positivo": 0, "Negativo": 0, "Neutro": 0}

# --- INTERFACCIA LATERALE (SIDEBAR) ---
with st.sidebar:
    st.title("Configurazione")
    # Viene richiesto uno slider per la temperatura (simulata in questo contesto)
    # Questo parametro influenzerebbe la stocasticità di un modello linguistico reale.
    temperatura = st.slider("Temperatura del Modello", 0.0, 1.0, 0.5, step=0.1)
    
    st.markdown("---")
    # Visualizzazione del contatore globale salvato nello stato
    st.write(f"Analisi effettuate in questa sessione: {st.session_state.analysis_count}")

# --- AREA PRINCIPALE ---
st.title("Dashboard di Classificazione Sentiment")
st.markdown("Inserisci un testo per analizzarne la polarità e visualizzare le statistiche aggregate.")

# Punto 1: Input di testo tramite text_area
testo_input = st.text_area("Testo da analizzare", placeholder="Inserisci qui il tuo commento...")

# Punto 3: Pulsante di analisi
if st.button("Analizza Sentiment"):
    if testo_input.strip() == "":
        st.warning("Per favore, inserisci del testo prima di procedere.")
    else:
        # Incremento del contatore totale nello stato della sessione
        st.session_state.analysis_count += 1
        
        # LOGICA DI CLASSIFICAZIONE (Simulata)
        # In un'applicazione reale, qui avverrebbe la chiamata al modello (es: LM Studio)
        # utilizzando la 'temperatura' definita nello slider.
        # In questo esercizio, simuliamo il risultato in base alla lunghezza del testo o parole chiave.
        
        testo_lower = testo_input.lower()
        if any(parola in testo_lower for parola in ["buono", "ottimo", "eccellente", "grazie", "bello"]):
            risultato = "Positivo"
        elif any(parola in testo_lower for parola in ["brutto", "scarso", "errore", "pessimo", "male"]):
            risultato = "Negativo"
        else:
            risultato = "Neutro"
        
        # Aggiornamento del database locale nello stato della sessione
        st.session_state.sentiment_storage[risultato] += 1
        
        # Mostra il risultato immediato
        st.success(f"Analisi completata. Sentiment rilevato: {risultato}")

# --- VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
st.subheader("Riepilogo Statistico")

# Creiamo un DataFrame dai dati in session_state per alimentare il grafico
df_risultati = pd.DataFrame(
    list(st.session_state.sentiment_storage.items()),
    columns=["Sentiment", "Conteggio"]
)

# Visualizzazione tramite grafico a barre nativo di Streamlit
# Questo grafico si aggiorna dinamicamente ogni volta che lo stato cambia
st.bar_chart(df_risultati.set_index("Sentiment"))

# Tabella di riepilogo per maggiore chiarezza dei dati
st.table(df_risultati)