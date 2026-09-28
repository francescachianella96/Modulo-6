import torch
from sentence_transformers import SentenceTransformer, util

# Inizializzazione della configurazione hardware
# Si verifica la presenza di una GPU compatibile con CUDA per accelerare i calcoli.
# In caso di assenza, il sistema utilizzerà automaticamente il processore centrale (CPU).
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Sistema in esecuzione su: {device.upper()}")

# Definizione della base di conoscenza
# Questa lista di stringhe contiene i documenti o le informazioni che il sistema può consultare.
knowledge_base = [
    "Il Quantum-Link 2026 utilizza cristalli di silicio per trasmettere dati a velocità luce.",
    "Per resettare il modulo Quantum-Link, tieni premuto il tasto blu per 5 secondi.",
    "La batteria del Quantum-Link dura circa 48 ore in modalità risparmio energetico.",
    "Il protocollo di sicurezza Q-Secure impedisce intercettazioni quantistiche esterne.",
    "Quantum-Link è compatibile solo con router di classe gravitazionale."
]

# Caricamento del modello di embedding
# 'all-MiniLM-L6-v2' è un modello pre-addestrato che trasforma frasi in vettori numerici (embedding).
# Specificando il parametro 'device', forziamo il caricamento del modello sulla risorsa hardware scelta.
model_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name, device=device)

# Codifica della base di conoscenza
# Il metodo encode trasforma la lista di testi in una matrice di vettori numerici.
# convert_to_tensor=True restituisce un tensore PyTorch caricato direttamente sulla GPU/CPU,
# ottimizzando le operazioni matematiche successive.
corpus_embeddings = embedder.encode(knowledge_base, convert_to_tensor=True)

def mini_rag_system(query):
    """
    Simula un sistema RAG (Retrieval-Augmented Generation).
    Prende una domanda dall'utente, trova il documento più pertinente e crea un prompt per un LLM.
    """
    print(f"\nDomanda dell'utente: {query}")
    
    # Fase 1: Recupero (Retrieval)
    # Trasformiamo la domanda dell'utente in un vettore numerico utilizzando lo stesso modello
    # usato per la base di conoscenza. Questo permette di confrontare i vettori nello stesso spazio vettoriale.
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Fase 2: Calcolo della similitudine
    # Utilizziamo la 'cosine similarity' per confrontare il vettore della domanda con tutti i vettori
    # della base di conoscenza. Il sistema restituisce un punteggio di affinità per ogni documento.
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Identifichiamo l'indice del documento con il punteggio più alto (il più simile alla domanda).
    best_idx = torch.argmax(cos_scores).item()
    
    # Estraiamo il testo corrispondente all'indice trovato.
    context = knowledge_base[best_idx]
    print(f"Documento identificato come pertinente: {context}")
    
    # Fase 3: Generazione aumentata (Augmentation)
    # Costruiamo una stringa che include sia il contesto recuperato che la domanda originale.
    # Questo prompt "arricchito" serve a guidare un modello linguistico (LLM) verso la risposta corretta.
    augmented_prompt = f"""
    Basati esclusivamente sulle seguenti informazioni per rispondere.
    Contesto: {context}
    Domanda: {query}
    Risposta:"""
    
    return augmented_prompt

# Esempio pratico di utilizzo del sistema
# Viene inoltrata una domanda e ottenuto il prompt finale pronto per essere elaborato.
final_prompt = mini_rag_system("Come si fa il reset del dispositivo?")
print("\n--- Struttura del Prompt per l'LLM ---")
print(final_prompt)