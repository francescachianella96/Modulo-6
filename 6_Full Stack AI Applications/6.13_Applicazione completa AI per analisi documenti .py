import streamlit as st
import fitz  # PyMuPDF: libreria per manipolare file PDF (lettura, estrazione testo, ecc.)
import faiss  # Libreria efficiente sviluppata da Facebook per la ricerca di similarità vettoriale
import numpy as np  # Libreria standard per calcoli numerici e gestione array multidimensionali
import requests  # Libreria per effettuare chiamate HTTP alle API REST
import json # Libreria per gestire il formato dati JSON (usato nelle risposte API)
from typing import List, Dict, Tuple # Type hints per rendere il codice più leggibile e sicuro

# -----------------------------------------------------------------------------
# CONFIGURAZIONE INIZIALE
# -----------------------------------------------------------------------------

# Configurazione della pagina Streamlit.
# Questa deve essere sempre la prima istruzione eseguibile in uno script Streamlit.
# Impostiamo il layout 'wide' per sfruttare tutto lo schermo per visualizzare i risultati.
st.set_page_config(layout="wide", page_title="Document Intelligence AI")

# Parametri di connessione a LM Studio (Server AI Locale).
# LM Studio espone API compatibili con lo standard OpenAI sulla porta 1234 di default.
LM_STUDIO_ENDPOINT = "http://localhost:1234/v1"

# Definiamo i nomi dei modelli che ci aspettiamo di trovare caricati su LM Studio.
# Questi nomi devono corrispondere a quelli visualizzati nell'interfaccia "Local Server".
EMBEDDING_MODEL_NAME = "text-embedding-gemma-300m" # Modello per trasformare testo in vettori
CHAT_MODEL_NAME = "gpt-oss-20b"            # Modello LLM per generare le risposte (Chat)

# -----------------------------------------------------------------------------
# CLASSE GESTIONE COMUNICAZIONE AI
# -----------------------------------------------------------------------------

class LMStudioClient:
    """
    Questa classe gestisce tutta la comunicazione con il server AI locale (LM Studio).
    Il suo scopo è nascondere la complessità delle chiamate HTTP raw al resto del programma,
    offrendo metodi semplici come get_embedding() e chat_completion().
    """
    
    def __init__(self, base_url: str):
        """
        Il costruttore della classe (metodo __init__) viene chiamato quando creiamo
        un nuovo oggetto LMStudioClient. Qui salviamo l'URL base del server.
        """
        self.base_url = base_url

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Questo metodo trasforma una stringa di testo in un vettore numerico (embedding).
        L'embedding cattura il significato semantico della frase: frasi simili avranno
        vettori "vicini" nello spazio matematico.
        
        Args:
            text: La stringa di testo da convertire.
            
        Returns:
            Un array NumPy contenente numeri float (il vettore).
        """
        try:
            # Effettua una richiesta POST all'endpoint /embeddings.
            # Inviamo il nome del modello da usare e il testo da processare.
            payload = {
                "model": EMBEDDING_MODEL_NAME,
                "input": text.replace("\n", " ") # Rimuoviamo i newline per evitare problemi di formattazione
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Verifica se la richiesta ha avuto successo (codice 200 OK).
            # Se il server risponde con un errore (es. 404 o 500), solleva un'eccezione.
            response.raise_for_status()
            
            # Estrae il vettore dalla risposta JSON.
            # La struttura tipica è: {'data': [{'embedding': [0.1, 0.2, ...]}], ...}
            embedding_list = response.json()['data'][0]['embedding']
            
            # Converte la lista Python in un array NumPy per poterci fare calcoli matematici efficienti.
            # Usiamo float32 per risparmiare memoria rispetto al default float64.
            return np.array(embedding_list, dtype='float32')
            
        except requests.exceptions.RequestException as e:
            # Gestione specifica degli errori di rete o del server.
            st.error(f"Errore di connessione a LM Studio: {e}")
            # Ritorna un vettore di zeri come fallback per non far crashare l'app,
            # ma segnala chiaramente il problema all'utente.
            return np.zeros(1, dtype='float32') 

    def chat_completion(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """
        Invia la conversazione al modello LLM e gestisce la risposta, visualizzandola
        man mano che viene generata (streaming).
        
        Args:
            messages: Lista di dizionari con la cronologia (es. [{'role': 'user', 'content': 'ciao'}]).
            temperature: Parametro che controlla la creatività del modello (0.0 predecibile, 1.0 creativo).
            
        Returns:
            La risposta completa (stringa) generata dal modello.
        """
        try:
            # Prepariamo i dati da inviare al modello.
            payload = {
                "model": CHAT_MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
                "stream": True # Abilitiamo lo streaming per ricevere la risposta token per token
            }

            # Invia la richiesta HTTP POST all'endpoint /chat/completions.
            # stream=True mantiene la connessione aperta.
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                stream=True, 
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            full_response = ""
            # Creiamo un elemento vuoto nella UI che aggiorneremo man mano che arrivano i pezzi di testo.
            placeholder = st.empty() 
            
            # Iteriamo sulle righe di testo in arrivo dal server (Server-Sent Events).
            for line in response.iter_lines():
                if line:
                    # Decodifica i byte in stringa UTF-8
                    decoded_line = line.decode('utf-8').strip()
                    
                    # Il formato SSE inizia con "data: "
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:] # Rimuoviamo il prefisso "data: "
                        
                        # "[DONE]" è un segnale speciale che indica la fine dello stream
                        if data_str != "[DONE]":
                            try:
                                # Parsing del JSON parziale riceuto
                                data = json.loads(data_str)
                                # Estrazione del contenuto del delta (il nuovo token/parola)
                                delta = data['choices'][0]['delta']
                                
                                # A volte il delta è vuoto (es. messaggi di keep-alive), controlliamo 'content'
                                if "content" in delta:
                                    content = delta["content"]
                                    full_response += content
                                    # Aggiorniamo la UI aggiungendo un cursore visivo '|' per far capire che sta scrivendo
                                    placeholder.markdown(full_response + " |")
                                    
                            except json.JSONDecodeError:
                                # Se il JSON è corrotto, lo ignoriamo e passiamo al prossimo chunk
                                continue
            
            # Aggiornamento finale per rimuovere il cursore '|' e mostrare il testo pulito
            placeholder.markdown(full_response)
            return full_response

        except Exception as e:
            st.error(f"Errore durante la generazione della risposta: {e}")
            return "Mi dispiace, si è verificato un errore di comunicazione con il modello."

# -----------------------------------------------------------------------------
# FUNZIONI DI UTILITÀ (PROCESSING DEL DOCUMENTO)
# -----------------------------------------------------------------------------

def extract_text_from_pdf(file_stream) -> List[Tuple[str, int]]:
    """
    Legge il file PDF caricato ed estrae il testo pagina per pagina usando PyMuPDF.
    
    Args:
        file_stream: Il file aperto (oggetto BytesIO) proveniente dallo st.file_uploader.
        
    Returns:
        Una lista di tuple, dove ogni tupla contiene (testo_pagina, numero_pagina).
    """
    # Apre il documento PDF direttamente dalla memoria RAM senza salvarlo su disco.
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    text_pages = []
    
    # Itera su ogni pagina del documento
    for page_num, page in enumerate(doc):
        # Estrae il testo semplice dalla pagina
        text = page.get_text()
        
        # Se la pagina contiene testo (non è vuota o solo immagini), la aggiungiamo alla lista.
        # Aggiungiamo 1 al numero pagina perché per gli umani le pagine partono da 1.
        if text.strip(): 
            text_pages.append((text, page_num + 1))
            
    return text_pages

def recursive_text_splitter(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Algoritmo di suddivisione del testo (Chunking).
    Divide un testo lungo in frammenti più piccoli, cercando di non tagliare a metà frasi o paragrafi.
    
    È "ricorsivo" nel senso che prova diverse strategie di taglio in ordine di priorità:
    1. Paragrafi (doppio a capo)
    2. Righe (singolo a capo)
    3. Frasi (punto)
    4. Parole (spazio)
    
    Args:
        text: Il testo intero da dividere.
        chunk_size: La lunghezza target di ogni frammento.
        overlap: Il numero di caratteri che si ripetono tra un chunk e il successivo (per mantenere il contesto).
        
    Returns:
        Lista di stringhe (i frammenti risultanti).
    """
    chunks = []
    start = 0
    text_len = len(text)

    # Continua finché non abbiamo processato tutto il testo
    while start < text_len:
        # Calcoliamo dove finirebbe idealmente il chunk corrente
        end = start + chunk_size
        
        # Se siamo vicini alla fine del testo, prendiamo tutto ciò che resta
        if end >= text_len:
            chunks.append(text[start:])
            break
            
        # Ricerca del miglior punto di taglio all'indietro
        cut_point = -1
        
        # Proviamo a tagliare su un doppio a capo (fine paragrafo)
        if cut_point == -1:
            last_double_newline = text.rfind('\n\n', start, end)
            if last_double_newline != -1:
                cut_point = last_double_newline + 2 # +2 include i caratteri \n\n nel chunk corrente
        
        # Proviamo a tagliare su un a capo singolo (fine riga)
        if cut_point == -1:
            last_newline = text.rfind('\n', start, end)
            if last_newline != -1:
                cut_point = last_newline + 1
        
        # Proviamo a tagliare su un punto (fine frase)
        if cut_point == -1:
            last_period = text.rfind('. ', start, end)
            if last_period != -1:
                cut_point = last_period + 1
                
        # Proviamo a tagliare su uno spazio (fine parola)
        if cut_point == -1:
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                cut_point = last_space + 1
            else:
                # Caso peggiore: nessun separatore trovato, tagliamo brutalmente alla lunghezza massima
                cut_point = end 

        # Aggiungiamo il frammento trovato alla lista dei chunk
        chunks.append(text[start:cut_point])
        
        # Calcoliamo il nuovo punto di partenza per la prossima iterazione.
        # Torniamo indietro di 'overlap' caratteri per creare la sovrapposizione.
        start = cut_point - overlap 
        
        # Controllo di sicurezza: se l'overlap ci fa tornare indietro troppo (dietro all'inizio corrente),
        # avanziamo comunque per evitare loop infiniti.
        if start >= cut_point: 
            start = cut_point
            
    return chunks

# -----------------------------------------------------------------------------
# LOGICA PRINCIPALE (MAIN)
# -----------------------------------------------------------------------------

def main():
    # Intestazione visibile nella pagina
    st.title("Document Intelligence: Analisi Documentale con AI Locale")
    st.markdown("""
    Questa applicazione dimostra il funzionamento di una pipeline RAG (Retrieval Augmented Generation).
    Puoi caricare un PDF, il sistema lo indicizzerà vettorialmente, e potrai fare domande in linguaggio naturale.
    """)

    # --- FASE 1: SETUP DELL'INTERFACCIA LATERALE ---
    # Usiamo la sidebar per i controlli, lasciando l'area principale pulita per la chat.
    with st.sidebar:
        st.header("1. Caricamento Documento")
        # Widget standard di Streamlit per caricare file
        uploaded_file = st.file_uploader("Seleziona il PDF da analizzare", type="pdf")
        
        st.divider()
        st.header("Info Sistema")
        # Visualizziamo i nomi dei modelli in uso per trasparenza
        st.text(f"Modello Embedding:\n{EMBEDDING_MODEL_NAME}")
        st.text(f"Modello Chat:\n{CHAT_MODEL_NAME}")
        
        # Pulsante per resettare l'applicazione
        if st.button("Nuova Sessione"):
            st.session_state.clear() # Cancella tutti i dati memorizzati
            st.rerun() # Ricarica lo script dall'inizio

    # Creiamo l'istanza del client per comunicare con l'AI
    client = LMStudioClient(LM_STUDIO_ENDPOINT)

    # --- FASE 2: GESTIONE DEL DOCUMENTO (INGESTION) ---
    # Questo blocco viene eseguito solo se c'è un file caricato e se non è già stato processato.
    # Usiamo 'st.session_state' come memoria persistente tra un ricaricamento e l'altro della pagina.
    if uploaded_file and "vector_store" not in st.session_state:
        
        # Mostriamo un indicatore di caricamento
        with st.spinner("Elaborazione del documento in corso... attendere prego."):
            
            # 2.1 Estrazione del testo dal PDF
            raw_pages = extract_text_from_pdf(uploaded_file)
            st.session_state['doc_name'] = uploaded_file.name
            
            # 2.2 Suddivisione in chunk (Frammentazione)
            all_chunks = []
            metadata = [] # Lista parallela per conservare le informazioni di origine (numero pagina)
            
            # Mostriamo una barra di progresso
            progress_bar = st.progress(0, text="Inizio segmentazione del testo...")
            
            for content, page_num in raw_pages:
                chunks = recursive_text_splitter(content, chunk_size=800, overlap=100)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    metadata.append({"page": page_num, "text": chunk})
            
            # 2.3 Creazione degli Embedding e Indice Vettoriale
            if all_chunks:
                # Facciamo una prova di embedding per capire la dimensione esatta dei vettori (es. 768)
                sample_emb = client.get_embedding("test")
                vector_dimension = sample_emb.shape[0] 
                
                # Inizializziamo l'indice FAISS.
                # 'IndexFlatIP' calcola il prodotto scalare (Inner Product).
                index = faiss.IndexFlatIP(vector_dimension)
                
                # Processiamo i chunk a gruppi (batch) per efficienza e per aggiornare la barra di progresso
                vectors = []
                batch_size = 5 
                
                for i in range(0, len(all_chunks), batch_size):
                    # Calcolo percentuale completamento
                    pct = i / len(all_chunks)
                    progress_bar.progress(pct, text=f"Generazione vettori: {int(pct*100)}%")
                    
                    # Estrazione batch corrente
                    batch = all_chunks[i:i+batch_size]
                    
                    for text in batch:
                        vec = client.get_embedding(text)
                        
                        # Normalizzazione L2: rende il vettore unitario (lunghezza = 1).
                        # Questo passo è cruciale perché con vettori unitari, 
                        # il prodotto scalare coincide con la similarità del coseno.
                        faiss.normalize_L2(vec.reshape(1, -1)) 
                        
                        vectors.append(vec)
                
                # Se abbiamo ottenuto vettori validi, li aggiungiamo all'indice
                if vectors:
                    vectors_np = np.vstack(vectors) # Creiamo un'unica matrice gigante
                    index.add(vectors_np) # Carichiamo tutto in memoria FAISS
                    
                    # Salviamo i risultati nello stato della sessione per non dover rifare tutto al prossimo click
                    st.session_state['vector_store'] = index
                    st.session_state['metadata'] = metadata
                    st.session_state['chat_history'] = [] # Inizializziamo lista messaggi vuota
                    
                    st.success("Documento analizzato con successo. Puoi iniziare a fare domande.")
                else:
                    st.error("Errore: Nessun testo valido estratto per creare l'indice.")
            
            # Rimuoviamo la barra di progresso una volta finito
            progress_bar.empty()

    # --- FASE 3: INTERAZIONE UTENTE (RAG CHAT) ---
    # Mostriamo l'interfaccia di chat solo se abbiamo un documento indicizzato pronto all'uso.
    if "vector_store" in st.session_state:
        st.divider()
        st.subheader("Chat Interattiva")
        
        # Ridisegniamo la cronologia dei messaggi precedenti
        for msg in st.session_state['chat_history']:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input dell'utente
        if prompt := st.chat_input("Fai una domanda sul contenuto del PDF..."):
            
            # 3.1 Gestione input utente
            # Visualizziamo subito la domanda
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state['chat_history'].append({"role": "user", "content": prompt})
            
            # 3.2 Retrieval (Recupero informazioni rilevanti)
            # Trasformiamo la domanda dell'utente in un vettore
            query_vec = client.get_embedding(prompt)
            faiss.normalize_L2(query_vec.reshape(1, -1)) # Normalizziamo anche la query
            
            # Cerchiamo i 3 chunk più simili nel nostro indice FAISS
            k = 3
            D, I = st.session_state['vector_store'].search(query_vec.reshape(1, -1), k)
            
            # Costruiamo la stringa di contesto da passare all'LLM
            retrieved_context = ""
            sources_text_display = ""
            
            # I[0] contiene gli indici dei risultati trovati
            for idx, neighbor_i in enumerate(I[0]):
                if neighbor_i < len(st.session_state['metadata']):
                    meta = st.session_state['metadata'][neighbor_i]
                    chunk_text = meta['text']
                    page = meta['page']
                    
                    # Aggiungiamo il testo al contesto 'invisibile' per l'LLM
                    retrieved_context += f"[FRAMMENTO {idx+1} (Pagina {page})]:\n{chunk_text}\n\n"
                    
                    # Prepariamo un testo formattato per mostrare le fonti all'utente (opzionale)
                    sources_text_display += f"- **Pagina {page}**: _{chunk_text[:120]}..._\n"

            # Sezione Debug (Espandibile): Utile per l'apprendimento, mostra cosa "vede" l'AI
            with st.expander("Dettagli Tecnici: Contesto recuperato (RAG)"):
                st.text(retrieved_context)

            # 3.3 Generation (Generazione Risposta)
            with st.chat_message("assistant"):
                # System Prompt: Istruisce l'AI su come comportarsi
                system_prompt = """Sei un assistente AI specializzato. 
                Rispondi alla domanda basandoti SOLO sul contesto fornito di seguito.
                Non usare conoscenze esterne. Se non sai la risposta, dillo.
                Cita sempre il numero di pagina delle informazioni usate."""
                
                full_prompt = f"CONTESTO:\n{retrieved_context}\n\nDOMANDA:\n{prompt}"
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
                
                # Chiamata all'LLM per ottenere la risposta
                response_text = client.chat_completion(messages)
                
                # Aggiungiamo le fonti in calce alla risposta
                if sources_text_display:
                    st.markdown("---")
                    st.caption("Fonti utilizzate per questa risposta:")
                    st.markdown(sources_text_display)
                    st.session_state['chat_history'].append({
                        "role": "assistant", 
                        "content": response_text + f"\n\n---\n*Fonti:*\n{sources_text_display}"
                    })
                else:
                    st.session_state['chat_history'].append({"role": "assistant", "content": response_text})

    else:
        # Messaggio di benvenuto se non c'è ancora nessun file
        st.info("Per iniziare, usa il pannello laterale a sinistra per caricare un documento PDF.")

# Blocco standard per l'esecuzione dello script Python
if __name__ == "__main__":
    main()